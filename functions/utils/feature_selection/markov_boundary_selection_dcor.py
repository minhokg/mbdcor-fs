from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

from functions.utils.correlation_functions.partial_dcor import partial_dcor


def _screen_features(
    x_subset: np.ndarray,
    y: np.ndarray,
    selected_columns: np.ndarray,
    alpha: float,
    random_state: int,
) -> List[int]:
    """Phase 1: keep columns with marginal dependence."""
    selected: List[int] = []
    for j, feature_idx in enumerate(selected_columns):
        p = partial_dcor(x_subset[:, j], y, cond=None, random_state=random_state)
        if p < alpha:
            selected.append(int(feature_idx))
    return selected


def _is_conditionally_independent(
    x: np.ndarray,
    y: np.ndarray,
    j: int,
    others: Sequence[int],
    n_samples: int,
    alpha: float,
    random_state: int,
) -> bool:
    """Return True if X_j ⫫ Y | Z for some Z ⊆ others."""
    for ksize in range(len(others) + 1):
        for cond_set in combinations(others, ksize):
            z = x[:, cond_set] if cond_set else np.empty((n_samples, 0))
            p = partial_dcor(x[:, j], y, cond=z, random_state=random_state)
            if p >= alpha:
                return True
    return False


def _prune_features(
    x: np.ndarray,
    y: np.ndarray,
    selected: List[int],
    n_samples: int,
    alpha: float,
    random_state: int,
) -> List[int]:
    """Phase 2: iteratively remove features that become conditionally independent."""
    selected_set = set(selected)
    changed = True

    while changed:
        changed = False
        # iterate over a snapshot to allow removals
        for j in list(selected_set):
            others = [k for k in selected_set if k != j]
            if _is_conditionally_independent(
                x=x,
                y=y,
                j=j,
                others=others,
                n_samples=n_samples,
                alpha=alpha,
                random_state=random_state,
            ):
                selected_set.remove(j)
                changed = True

    return sorted(selected_set)


def _maybe_resample_schedule(
    step: int,
    n_resamples: int,
    n_samples: int,
    n_features: int,
    feature_count: np.ndarray,
    non_zero_columns: List[int],
) -> tuple[np.ndarray, List[int], int]:
    """
    Match your existing behavior.

    - Every int(n_samples/5) iterations, recompute non-zero columns,
      update sqrt-based selection count, and reset feature_count unless last step.
    """
    period = max(1, int(n_samples / 5))
    if (step % period) != 0:
        return feature_count, non_zero_columns, int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0

    non_zero_columns = [i for i, count in enumerate(feature_count) if count > 0]
    num_features_to_select = int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0

    if step != n_resamples:
        feature_count = np.zeros(n_features)

    return feature_count, non_zero_columns, num_features_to_select


def _final_selection(feature_count: np.ndarray) -> Dict[int, float]:
    """Phase 5: your dynamic threshold logic, with safe empty-handling."""
    non_zero_values = feature_count[feature_count > 0]
    if non_zero_values.size == 0:
        return {}

    std = float(np.std(non_zero_values))
    threshold_diff = 2.0 * std

    min_non_zero = float(np.min(non_zero_values))
    if min_non_zero > threshold_diff:
        return {int(i): float(c) for i, c in enumerate(feature_count) if c > 0}

    median_non_zero = float(np.median(non_zero_values))
    return {int(i): float(c) for i, c in enumerate(feature_count) if c > median_non_zero}


def markov_boundary_selection_dcor(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    n_resamples: int = 500,
    random_state: int = 42,
) -> Dict[int, float]:
    """
    Estimate the Markov Boundary of a target variable using partial distance correlation.

    Returns:
        Dict[int, float]: mapping feature index -> count

    """
    n_samples, n_features = x.shape
    feature_count = np.zeros(n_features)

    rng = np.random.default_rng(random_state)
    non_zero_columns: List[int] = list(range(n_features))
    num_features_to_select = int(np.sqrt(n_features))

    for n in range(n_resamples):
        if num_features_to_select <= 0 or len(non_zero_columns) == 0:
            break

        selected_columns = rng.choice(
            non_zero_columns,
            size=min(num_features_to_select, len(non_zero_columns)),
            replace=False,
        )

        x_subset = x[:, selected_columns]
        selected = _screen_features(
            x_subset=x_subset,
            y=y,
            selected_columns=selected_columns,
            alpha=alpha,
            random_state=random_state,
        )

        selected = _prune_features(
            x=x,
            y=y,
            selected=selected,
            n_samples=n_samples,
            alpha=alpha,
            random_state=random_state,
        )

        for feature in selected:
            feature_count[feature] += 1

        feature_count, non_zero_columns, num_features_to_select = _maybe_resample_schedule(
            step=n + 1,
            n_resamples=n_resamples,
            n_samples=n_samples,
            n_features=n_features,
            feature_count=feature_count,
            non_zero_columns=non_zero_columns,
        )

    return _final_selection(feature_count)
