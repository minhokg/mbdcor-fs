import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from mbdcor_fs.utils.feature_selection.boruta_selection_classifier import boruta_selection_classifier
from mbdcor_fs.utils.feature_selection.markov_boundary_selection_dcor import markov_boundary_selection_dcor
from mbdcor_fs.utils.helper.setup_logging import setup_logging
from mbdcor_fs.utils.train_evaluate.train_evaluate_xgboost_classifier import train_evaluate_xgboost_classifier


def load_wdbc(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the WDBC dataset.

    This function reads the raw ``wdbc.data`` file, removes the identifier column,
    encodes the diagnosis label into binary format, and standardizes the features.

    :param path: Path to the ``wdbc.data`` file.

    :return: Tuple containing the feature matrix ``X`` and target vector ``y``.
    """
    columns = ["id", "diagnosis"] + [f"f{i}" for i in range(30)]
    df = pd.read_csv(path, header=None, names=columns)

    # drop ID
    df = df.drop(columns=["id"])

    # encode target
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    x = df.drop(columns=["diagnosis"]).values
    y = df["diagnosis"].values

    # scale features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y


def _bootstrap_train_oob_test_split(
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw bootstrap training indices and return the out-of-bootstrap test indices."""
    train_idx = rng.choice(n_samples, size=n_samples, replace=True)

    in_bag_mask = np.zeros(n_samples, dtype=bool)
    in_bag_mask[train_idx] = True
    test_idx = np.where(~in_bag_mask)[0]

    return train_idx, test_idx


def _run_one_real(
    x: np.ndarray,
    y: np.ndarray,
    sim: int,
    alpha_mb: float = 0.05,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Execute one Monte Carlo iteration using bootstrap sampling.

    This function performs bootstrap resampling on the dataset, applies
    feature selection using Boruta and MBDcor, and evaluates the selected
    features using an XGBoost classifier.

    :param x: Feature matrix of shape (n_samples, n_features).
    :param y: Binary target vector of shape (n_samples,).
    :param sim: Simulation index used for reproducibility.
    :param alpha_mb: Significance level for MBDcor feature selection.
    :param random_state: Base random seed.

    :return: List of result dictionaries for each method.
    """
    rng = np.random.default_rng(random_state + sim)

    # -------------------------
    # Bootstrap sampling
    # -------------------------
    train_idx, test_idx = _bootstrap_train_oob_test_split(n_samples=len(y), rng=rng)

    if len(test_idx) == 0:
        return []

    x_train_raw = x[train_idx]
    y_train = y[train_idx]
    x_test_raw = x[test_idx]
    y_test = y[test_idx]

    # fit scaler on training only
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)

    results: List[Dict[str, Any]] = []

    # -------------------------
    # Boruta
    # -------------------------
    t0 = time.perf_counter()
    boruta_sel = boruta_selection_classifier(x=x_train, y=y_train)
    t1 = time.perf_counter()

    if len(boruta_sel) > 0:
        log_loss_boruta = train_evaluate_xgboost_classifier(x_train=x_train[:, boruta_sel], y_train=y_train, x_test=x_test[:, boruta_sel], y_test=y_test, random_state=random_state + sim)

        results.append(
            {
                "method": "Boruta",
                "sim": sim,
                "runtime_s": t1 - t0,
                "n_selected": len(boruta_sel),
                "log_loss": log_loss_boruta,
            }
        )

    # -------------------------
    # MBDcor
    # -------------------------
    t0 = time.perf_counter()
    mb_sel = markov_boundary_selection_dcor(
        x=x_train,
        y=y_train,
        alpha=alpha_mb,
        random_state=random_state + sim,
    )
    t1 = time.perf_counter()

    if len(mb_sel) > 0:
        log_loss_mbdcor = train_evaluate_xgboost_classifier(
            x_train=x_train[:, mb_sel],
            y_train=y_train,
            x_test=x_test[:, mb_sel],
            y_test=y_test,
            random_state=random_state + sim,
        )

        results.append(
            {
                "method": "MBDcor",
                "sim": sim,
                "runtime_s": t1 - t0,
                "n_selected": len(mb_sel),
                "log_loss": log_loss_mbdcor,
            }
        )

    return results


def real_data_experiment(
    x: np.ndarray,
    y: np.ndarray,
    n_sims: int = 50,
    alpha_mb: float = 0.05,
    max_workers: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Monte Carlo experiments on real data using bootstrap resampling.

    This function executes multiple simulations in parallel, where each
    simulation applies feature selection and model evaluation on a bootstrap
    sample of the dataset. The results are aggregated to compute summary
    statistics for each method.

    :param x: Feature matrix of shape (n_samples, n_features).
    :param y: Binary target vector of shape (n_samples,).
    :param n_sims: Number of Monte Carlo simulations.
    :param alpha_mb: Significance level for MBDcor feature selection.
    :param max_workers: Maximum number of parallel workers. If ``None``, uses default.

    :return: Tuple containing:
             - raw results DataFrame
             - aggregated summary DataFrame
    """
    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one_real, x, y, sim, alpha_mb) for sim in range(n_sims)]

        for fut in as_completed(futures):
            all_rows.extend(fut.result())

    raw_df = pd.DataFrame(all_rows)

    summary_df = raw_df.groupby("method", as_index=False).agg(
        log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"),
        time_mean=("runtime_s", "mean"),
        time_std=("runtime_s", "std"),
        nsel_mean=("n_selected", "mean"),
        nsel_std=("n_selected", "std"),
    )

    return raw_df, summary_df


# run wisconsin
if __name__ == "__main__":
    setup_logging()

    x, y = load_wdbc("data/wdbc.data")
    raw, summary = real_data_experiment(
        x=x,
        y=y,
        n_sims=100,
    )

    logging.info(summary)
    raw.to_csv("results/real_data_simulation/raw_wbdc.csv", index=False)
    summary.to_csv("results/real_data_simulation/summary_wbdc.csv", index=False)
