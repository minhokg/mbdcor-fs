import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    return x, y


def _run_one_real_data_experiment(
    x: np.ndarray,
    y: np.ndarray,
    sim: int,
    alpha_mb: float = 0.05,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Execute one Monte Carlo iteration using a random train/test split.

    This function splits the dataset into training and test sets, applies
    feature selection using Boruta and MBDcor on the training data, and
    evaluates the selected features using an XGBoost classifier on the
    held-out test data.

    :param x: Feature matrix of shape (n_samples, n_features).
    :param y: Binary target vector of shape (n_samples,).
    :param sim: Simulation index used for reproducibility.
    :param alpha_mb: Significance level for MBDcor feature selection.
    :param random_state: Base random seed.

    :return: List of result dictionaries for each method.
    """
    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state + sim)

    # scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results: List[Dict[str, Any]] = []
    # -------------------------
    # Boruta
    # -------------------------
    t0 = time.perf_counter()
    boruta_sel_list = boruta_selection_classifier(x_train=x_train, y_train=y_train)
    t1 = time.perf_counter()

    if len(boruta_sel_list) > 0:
        log_loss_boruta = train_evaluate_xgboost_classifier(x_train=x_train[:, boruta_sel_list], y_train=y_train, x_test=x_test[:, boruta_sel_list], y_test=y_test, random_state=random_state + sim)

        results.append(
            {
                "method": "Boruta",
                "sim": sim,
                "runtime_s": t1 - t0,
                "n_selected": len(boruta_sel_list),
                "log_loss": log_loss_boruta,
            }
        )

    # -------------------------
    # MBDcor
    # -------------------------
    t0 = time.perf_counter()
    mb_sel_list = markov_boundary_selection_dcor(
        x=x_train,
        y=y_train,
        alpha=alpha_mb,
        random_state=random_state + sim,
    )
    t1 = time.perf_counter()

    if len(mb_sel_list) > 0:
        log_loss_mbdcor = train_evaluate_xgboost_classifier(
            x_train=x_train[:, mb_sel_list],
            y_train=y_train,
            x_test=x_test[:, mb_sel_list],
            y_test=y_test,
            random_state=random_state + sim,
        )

        results.append(
            {
                "method": "MBDcor",
                "sim": sim,
                "runtime_s": t1 - t0,
                "n_selected": len(mb_sel_list),
                "log_loss": log_loss_mbdcor,
            }
        )

    return results


def real_data_experiment(
    x: np.ndarray,
    y: np.ndarray,
    n_sims: int = 50,
    alpha_mb: float = 0.01,
    max_workers: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run repeated Monte Carlo experiments on real data using random train/test splits.

    This function executes multiple simulations in parallel. In each
    simulation, the dataset is randomly split into training and test sets,
    feature selection is applied on the training data, and predictive
    performance is evaluated on the test data.

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

    logging.info(f"Starting real data experiment with {n_sims} simulations...")

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_run_one_real_data_experiment, x, y, sim, alpha_mb) for sim in range(n_sims)]

        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            all_rows.extend(result)
            logging.info(f"Completed simulation {i}/{n_sims}")

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
    root_path = "results/real_data_simulation"
    os.makedirs(root_path, exist_ok=True)
    raw.to_csv(root_path + "/raw_wbdc.csv", index=False)
    summary.to_csv(root_path + "/summary_wbdc.csv", index=False)
