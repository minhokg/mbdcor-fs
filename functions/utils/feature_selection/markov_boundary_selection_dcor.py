from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

from functions.utils.correlation_functions.partial_dcor import partial_dcor


def _get_marginal_dependent_features(
    x_subset: np.ndarray,
    y: np.ndarray,
    selected_columns: np.ndarray,
    alpha: float,
    random_state: int,
) -> List[int]:
    """
    Get features that exhibit marginal dependence with the target variable.

    Phase 1: compute (partial) distance correlation p-values for each feature
    in ``x_subset`` against ``y`` (no conditioning set) and keep those with
    ``p < alpha``.

    :param x_subset: Submatrix of selected features with shape
                     (n_samples, len(selected_columns)).
    :param y: Target variable of shape (n_samples,).
    :param selected_columns: Array of original feature indices
                             corresponding to columns in x_subset.
    :param alpha: Significance level for partial distance
                  correlation tests.
    :param random_state: Seed used inside partial distance
                         correlation testing.
    :return: List of original feature indices that show
             statistically significant marginal dependence
             with the target.
    """
    selected: List[int] = []
    for j, feature_idx in enumerate(selected_columns):
        p = partial_dcor(x_subset[:, j], y, cond=None, random_state=random_state)
        if p < alpha:
            selected.append(int(feature_idx))
    return selected


def _test_conditional_independence(
    x: np.ndarray,
    y: np.ndarray,
    j: int,
    others: Sequence[int],
    n_samples: int,
    alpha: float,
    random_state: int,
) -> bool:
    """
    Test whether ``X_j`` is conditionally independent of ``Y``.

    This searches over all subsets of ``others`` and returns ``True`` if
    conditional independence is detected for at least one conditioning set.

    :param x: Full feature matrix of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples,).
    :param j: Index of the feature being tested.
    :param others: Sequence of indices corresponding to the
                   remaining selected features.
    :param n_samples: Number of samples in the dataset.
    :param alpha: Significance level for partial distance
                  correlation tests.
    :param random_state: Seed used inside partial distance
                         correlation testing.
    :return: True if ``X_j`` is conditionally independent of ``Y``
             given some subset of others; otherwise False.
    """
    for ksize in range(len(others) + 1):
        for cond_set in combinations(others, ksize):
            z = x[:, cond_set] if cond_set else np.empty((n_samples, 0))
            p = partial_dcor(x[:, j], y, cond=z, random_state=random_state)
            if p >= alpha:
                return True
    return False


def _remove_conditional_independent_features(
    x: np.ndarray,
    y: np.ndarray,
    selected: List[int],
    n_samples: int,
    alpha: float,
    random_state: int,
) -> List[int]:
    """
    Remove features that are conditionally independent of the target.

    Phase 2: iteratively remove any feature that becomes conditionally
    independent of ``y`` given the remaining selected features, until no
    further removals are possible.

    :param x: Full feature matrix of shape (n_samples, n_features).
    :param y: Target variable of shape (n_samples,).
    :param selected: List of feature indices retained after
                     the screening phase.
    :param n_samples: Number of samples in the dataset.
    :param alpha: Significance level for partial distance
                  correlation tests.
    :param random_state: Seed used inside partial distance
                         correlation testing.
    :return: Sorted list of feature indices that remain after
             iterative pruning.
    """
    selected_set = set(selected)
    changed = True

    while changed:
        changed = False
        for j in list(selected_set):
            others = [k for k in selected_set if k != j]
            if _test_conditional_independence(
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


def _update_resample_schedule(
    step: int,
    n_resamples: int,
    n_samples: int,
    n_features: int,
    feature_count: np.ndarray,
    non_zero_columns: List[int],
) -> tuple[np.ndarray, List[int], int]:
    """
    Update the resampling schedule at predefined intervals.

    Every ``int(n_samples / 5)`` iterations this recomputes the non-zero
    feature set, updates the number of features to sample using a square-root
    rule, and resets counts unless at the final iteration.

    :param step: Current resampling iteration (1-indexed).
    :param n_resamples: Total number of resampling iterations.
    :param n_samples: Number of samples in the dataset.
    :param n_features: Total number of features.
    :param feature_count: Array of length n_features containing
                          current selection counts.
    :param non_zero_columns: List of feature indices with
                             non-zero selection counts.
    :return: Tuple containing:
             - Updated feature_count array,
             - Updated list of non-zero feature indices,
             - Updated number of features to select in the
               next iteration.
    """
    period = max(1, int(n_samples / 5))
    if (step % period) != 0:
        return (
            feature_count,
            non_zero_columns,
            int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0,
        )

    non_zero_columns = [i for i, count in enumerate(feature_count) if count > 0]
    num_features_to_select = int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0

    if step != n_resamples:
        feature_count = np.zeros(n_features)

    return feature_count, non_zero_columns, num_features_to_select


def _final_selection(feature_count: np.ndarray) -> Dict[int, float]:
    """
    Select final features via dynamic thresholding.

    Phase 5: if the minimum non-zero count exceeds twice the standard
    deviation, all non-zero features are kept. Otherwise, features above the
    median non-zero count are retained.

    :param feature_count: Array of selection counts for
                          each feature.
    :return: Dictionary mapping selected feature indices
             to their final selection counts. Returns an
             empty dictionary if all counts are zero.
    """
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
    Estimate a Markov boundary using partial distance correlation.

    The algorithm repeatedly samples subsets of features, identifies those
    marginally dependent on the target using distance correlation, removes
    conditionally independent features, and aggregates selection frequencies
    across resamples. The final output retains features based on their
    stability across iterations.

    :param x: Feature matrix of shape ``(n_samples, n_features)``, where
              each column represents a candidate predictor.
    :param y: Target variable of shape ``(n_samples,)``.
    :param alpha: Significance level used in (partial) distance correlation
                  independence tests. Default is ``0.01``.
    :param n_resamples: Number of resampling iterations used to estimate
                        feature stability. Default is ``500``.
    :param random_state: Seed for the random number generator to ensure
                         reproducibility. Default is ``42``.
    :returns: A dictionary mapping feature indices to their selection
              frequencies across resamples, after applying the final
              selection criterion.
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
        selected = _get_marginal_dependent_features(
            x_subset=x_subset,
            y=y,
            selected_columns=selected_columns,
            alpha=alpha,
            random_state=random_state,
        )

        selected = _remove_conditional_independent_features(
            x=x,
            y=y,
            selected=selected,
            n_samples=n_samples,
            alpha=alpha,
            random_state=random_state,
        )

        for feature in selected:
            feature_count[feature] += 1

        feature_count, non_zero_columns, num_features_to_select = _update_resample_schedule(
            step=n + 1,
            n_resamples=n_resamples,
            n_samples=n_samples,
            n_features=n_features,
            feature_count=feature_count,
            non_zero_columns=non_zero_columns,
        )

    return _final_selection(feature_count)
