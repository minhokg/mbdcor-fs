from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functions.utils.base.models import RunResult
from functions.utils.feature_selection.boruta_selection_classifier import boruta_selection_classifier
from functions.utils.feature_selection.markov_boundary_selection_dcor import markov_boundary_selection_dcor
from functions.utils.train_evaluate.train_evaluate_xgboost_classifier import train_evaluate_xgboost_classifier


def synthetic_simulation(
    p_list: Sequence[int] = (50, 100, 200, 300, 500),
    n_sims: int = 100,
    alpha_mb: float = 0.05,
    max_workers: int | None = None,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the full Monte Carlo experiment in parallel with varying 'n'."""
    rng = np.random.default_rng(random_seed)

    # build list of simulation tasks
    # each task corresponds to one (p, simulation replicate) pair
    tasks: list[tuple[int, int]] = [(p, sim) for p in p_list if p >= 6 for sim in range(n_sims)]

    # pre-generate n values deterministically
    n_values = rng.integers(500, 1001, size=len(tasks))

    all_rows: List[RunResult] = []

    # run simulations in parallel
    # each worker executes one configuration via `_run_one_setting`
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _run_one_setting,
                int(n_values[i]),
                p,
                sim,
                alpha_mb,
                random_seed,
            )
            for i, (p, sim) in enumerate(tasks)
        ]

        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    raw_df = pd.DataFrame([r.__dict__ for r in all_rows])

    # aggregate simulation statistics per method and feature dimension
    summary_df = (
        raw_df.groupby(["method", "p"], as_index=False)
        .agg(
            recall_mean=("recall_pct", "mean"),
            recall_std=("recall_pct", "std"),
            time_mean=("runtime_s", "mean"),
            time_std=("runtime_s", "std"),
            nsel_mean=("n_selected", "mean"),
            nsel_std=("n_selected", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
        .sort_values(["p", "method"])
        .reset_index(drop=True)
    )

    return raw_df, summary_df


def _make_spd_from_corr(
    corr: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Convert a symmetric matrix into a symmetric positive definite (SPD) matrix.

    This function ensures that the input matrix can be used for Cholesky
    decomposition by applying eigenvalue clipping and diagonal jitter.

    :param corr: Square correlation-like matrix.
    :param eps: Small positive constant used for eigenvalue clipping and
                numerical stabilization.
    :return: Symmetric positive definite matrix suitable for Cholesky decomposition.
    """
    # ensure symmetry
    a = (corr + corr.T) / 2.0

    # eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(a)

    # clip negative eigenvalues
    eigenvalues = np.clip(eigenvalues, eps, None)

    # reconstruct spd matrix
    a_spd = (eigenvectors * eigenvalues) @ eigenvectors.T

    # normalize diagonal to 1 to maintain correlation structure
    diag_vals = np.sqrt(np.diag(a_spd))
    a_spd = a_spd / np.outer(diag_vals, diag_vals)

    # numerical stabilization
    a_spd = (a_spd + a_spd.T) / 2.0
    a_spd += np.eye(a_spd.shape[0]) * eps

    return a_spd


def _generate_corr_matrix(
    p: int,
    rng: np.random.Generator,
    base_pairs: Sequence[Tuple[int, int, float]] = (
        (0, 1, 0.8),
        (2, 3, 0.5),
        (4, 5, 0.3),
    ),
    extra_pairs_frac: float = 0.01,
    extra_rho_range: Tuple[float, float] = (0.05, 0.25),
) -> np.ndarray:
    """
    Generate a correlation matrix with structured and sparse random correlations.

    The matrix starts as identity, enforces correlations on specified
    feature pairs, adds sparse random correlations, and is projected
    to SPD form.

    :param p: Number of features.
    :param rng: Random number generator.
    :param base_pairs: Feature index pairs with fixed correlation values.
    :param extra_pairs_frac: Fraction of feature pairs assigned additional
                             random correlations.
    :param extra_rho_range: Range for magnitude of additional correlations.
    :return: Symmetric positive definite correlation matrix.
    """
    corr = np.eye(p, dtype=float)

    # insert predefined correlations among some feature pairs
    for i, j, rho in base_pairs:
        if i < p and j < p:
            corr[i, j] = rho
            corr[j, i] = rho

    # add additional sparse correlations to mimic real datasets
    n_possible = p * (p - 1) // 2
    n_extra = int(extra_pairs_frac * n_possible)

    if n_extra > 0:
        added = 0
        while added < n_extra:
            i = int(rng.integers(0, p))
            j = int(rng.integers(0, p))
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if corr[i, j] != 0:
                continue
            rho = float(rng.uniform(*extra_rho_range))
            rho *= -1.0 if rng.random() < 0.5 else 1.0
            corr[i, j] = rho
            corr[j, i] = rho
            added += 1

    return _make_spd_from_corr(corr)


def _generate_data(
    n: int,
    p: int,
    rng: np.random.Generator,
    n_true: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic binary classification dataset with correlated features.

    The true variable is constructed using a randomly selected subset
    of features and random nonlinear transformations.

    :param n: Number of samples.
    :param p: Number of features.
    :param rng: Random number generator.
    :param n_true: Number of true variables.
    :return: Tuple (X, y, true_features) where
             X is the feature matrix,
             y is the binary target,
             and true_features contains indices of predictive features.
    """
    # generate correlated feature matrix
    corr = _generate_corr_matrix(p=p, rng=rng)
    chol = np.linalg.cholesky(corr)

    x_uncorr = rng.normal(size=(n, p))
    x = x_uncorr @ chol.T

    # select true feature indices
    true_features = rng.choice(p, size=n_true, replace=False)
    true_features = np.sort(true_features)

    # random nonlinear transformation generator
    def random_transform(x_col: np.ndarray) -> np.ndarray:
        transform_type = rng.choice(
            [
                "linear",
                "square",
                "cube",
                "sin",
                "cos",
                "exp",
                "log",
                "tanh",
            ]
        )

        if transform_type == "square":
            return x_col**2
        elif transform_type == "cube":
            return x_col**3
        elif transform_type == "sin":
            return np.sin(x_col)
        elif transform_type == "cos":
            return np.cos(x_col)
        elif transform_type == "exp":
            return np.exp(x_col)
        elif transform_type == "log":
            return np.log(np.abs(x_col) + 1.0)
        elif transform_type == "tanh":
            return np.tanh(x_col)
        else:
            return x_col

    # construct latent signal
    fx = np.zeros(n)

    for f in true_features:
        coef = rng.uniform(0.5, 3.0)
        sign = rng.choice([-1.0, 1.0])
        fx += sign * coef * random_transform(x[:, f])

    # add stochastic noise
    noise_scale = rng.uniform(0.5, 1.5)
    fx += rng.normal(scale=noise_scale, size=n)

    # logistic transformation
    probs = 1.0 / (1.0 + np.exp(-fx))
    y = (probs > 0.5).astype(int)

    return x, y, true_features


def _recall_percent(selected, truth):
    """
    Compute recall percentage of selected features.

    :param selected: Selected feature indices.
    :param truth: Ground truth feature indices.
    :return: Recall expressed as a percentage.
    """
    sel = np.array(list(selected))
    if len(sel) == 0:
        return 0.0
    return 100.0 * np.intersect1d(sel, truth).size / len(truth)


def _run_one_setting(
    n: int,
    p: int,
    sim: int,
    alpha_mb: float,
    random_state: int = 42,
) -> List[RunResult]:
    """
    Run a single Monte Carlo simulation comparing Boruta and MBDcor.

    :param n: Number of samples.
    :param p: Number of features.
    :param sim: Simulation replicate index.
    :param alpha_mb: Significance level for MBDcor.
    :param random_state: Base random seed.
    :return: List containing results for both methods.
    """
    rng = np.random.default_rng(random_state)

    x, y, true_features = _generate_data(n=n, p=p, rng=rng)

    out: List[RunResult] = []

    # boruta feature selection
    t0 = time.perf_counter()
    boruta_sel_list = boruta_selection_classifier(x=x, y=y)
    t1 = time.perf_counter()
    if len(boruta_sel_list) > 0:
        log_loss_boruta = train_evaluate_xgboost_classifier(x=x[:, boruta_sel_list], y=y)

        out.append(
            RunResult(
                method="Boruta",
                n=n,
                p=p,
                sim=sim,
                runtime_s=t1 - t0,
                recall_pct=_recall_percent(boruta_sel_list, true_features),
                n_selected=len(boruta_sel_list),
                log_loss=log_loss_boruta,
            )
        )

    # mbdcor feature selection
    t0 = time.perf_counter()
    mb_sel_list = markov_boundary_selection_dcor(
        x=x,
        y=y.ravel(),
        alpha=alpha_mb,
        random_state=random_state,
    )
    t1 = time.perf_counter()
    if len(mb_sel_list) > 0:
        log_loss_mbdcor = train_evaluate_xgboost_classifier(x=x[:, mb_sel_list], y=y)

        out.append(
            RunResult(
                method="MBDcor",
                n=n,
                p=p,
                sim=sim,
                runtime_s=t1 - t0,
                recall_pct=_recall_percent(mb_sel_list, true_features),
                n_selected=len(mb_sel_list),
                log_loss=log_loss_mbdcor,
            )
        )

    return out


def plot_runtime(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot the relationship between the number of parameters and the mean runtime for each feature selection method.

    :param summary_df: DataFrame containing aggregated simulation results.
    :param save_path: Path to save the figure.

    :return: None.
    """
    plt.figure(figsize=(8, 5))

    for method, df_method in summary_df.groupby("method"):
        plt.errorbar(
            df_method["p"],
            df_method["time_mean"],
            yerr=df_method["time_std"],
            marker="o",
            label=method,
            capsize=4,
            linestyle="-",
        )

    plt.xlabel("Number of Parameters (p)")
    plt.ylabel("Mean Runtime (seconds)")
    plt.title("Runtime Comparison: Boruta vs MBDcor")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_logloss(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot the relationship between the number of parameters and the mean log loss for each feature selection method.

    :param summary_df: DataFrame containing aggregated simulation results.
    :param save_path: Path to save the plot to.
    :return: None.
    """
    plt.figure(figsize=(8, 5))

    for method, df_method in summary_df.groupby("method"):
        plt.errorbar(
            df_method["p"],
            df_method["log_loss_mean"],
            yerr=df_method["log_loss_std"],  # <- error bars for standard deviation
            marker="o",
            label=method,
            capsize=4,  # small line at the end of error bars
            linestyle="-",  # solid line connecting points
        )

    plt.xlabel("Number of Parameters (p)")
    plt.ylabel("Mean Log Loss")
    plt.title("Prediction Performance: Boruta vs MBDcor")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_nsel(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot the relationship between the number of parameters and the mean number of selected features for each feature selection method, including standard deviation as error bars.

    :param summary_df: DataFrame containing aggregated simulation results.
    :param save_path: Path to save the plot to.
    :return: None.
    """
    plt.figure(figsize=(8, 5))

    for method, df_method in summary_df.groupby("method"):
        plt.errorbar(
            df_method["p"],
            df_method["nsel_mean"],
            yerr=df_method["nsel_std"],  # error bars
            marker="o",
            label=method,
            capsize=4,
            linestyle="-",
        )

    plt.xlabel("Number of Parameters (p)")
    plt.ylabel("Mean Number of Selected Features")
    plt.title("Feature Selection: Number of Selected Features (Mean ± Std)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_recall(summary_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot the relationship between the number of parameters and the mean recall for each feature selection method, including standard deviation as error bars.

    :param summary_df: DataFrame containing aggregated simulation results.
    :param save_path: Path to save the plot to.
    :return: None.
    """
    plt.figure(figsize=(8, 5))

    for method, df_method in summary_df.groupby("method"):
        plt.errorbar(
            df_method["p"],
            df_method["recall_mean"],
            yerr=df_method["recall_std"],  # error bars
            marker="o",
            label=method,
            capsize=4,
            linestyle="-",
        )

    plt.xlabel("Number of Parameters (p)")
    plt.ylabel("Mean Recall (%)")
    plt.title("Feature Selection Recall (Mean ± Std)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    raw, summary = synthetic_simulation(
        p_list=[50, 100, 150, 200, 250],
        n_sims=100,
    )
    os.makedirs("results", exist_ok=True)
    raw.to_csv("results/summary_results_error_bar_2.csv", index=False)
    summary.to_csv("results/summary_results_error_bar_2" ".csv", index=False)
    plot_runtime(summary, save_path="results/runtime_vs_p_error_bar_2.png")
    plot_logloss(summary, save_path="results/logloss_vs_p_error_bar_2.png")
    plot_nsel(summary, save_path="results/nsel_vs_p_2.png")
    plot_recall(summary, save_path="results/recall_vs_p_2.png")
