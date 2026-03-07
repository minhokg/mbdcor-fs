from __future__ import annotations

from typing import List

import numpy as np

from functions.utils.correlation_functions.partial_dcor import partial_dcor


def markov_boundary_selection_dcor(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    n_resamples: int = 500,
    random_state: int = 42,
) -> List[int]:
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

    marginal_pvals = np.array([partial_dcor(x[:, j], y, cond=None) for j in range(n_features)])

    for n in range(n_resamples):
        if num_features_to_select <= 0 or len(non_zero_columns) == 0:
            break

        selected_columns = rng.choice(
            non_zero_columns,
            size=min(num_features_to_select, len(non_zero_columns)),
            replace=False,
        )

        selected = _get_marginal_dependent_features(
            marginal_pvals=marginal_pvals,
            selected_columns=selected_columns,
            alpha=alpha,
        )

        selected = _remove_conditional_independent_features(
            x=x,
            y=y,
            selected=selected,
            alpha=alpha,
        )

        feature_count[selected] += 1

        feature_count, non_zero_columns, num_features_to_select = _update_resample_schedule(
            step=n + 1,
            n_resamples=n_resamples,
            feature_count=feature_count,
            non_zero_columns=non_zero_columns,
        )

    return _final_selection(feature_count)


def _benjamini_hochberg(p_values, alpha):
    """
    Perform Benjamini–Hochberg False Discovery Rate (FDR) correction.

    This procedure controls the expected proportion of false discoveries
    (Type I errors) among the rejected hypotheses.

    :param p_values: Array-like sequence of p-values corresponding to
                     multiple hypothesis tests.
    :param alpha: Desired false discovery rate level (e.g., 0.05).
    :return: Boolean NumPy array indicating which hypotheses are rejected
             after applying the Benjamini–Hochberg correction.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = sorted_p <= thresholds
    rejected = np.zeros(m, dtype=bool)
    if np.any(below):
        max_idx = np.max(np.where(below))
        rejected[sorted_idx[: max_idx + 1]] = True
    return rejected


def _get_marginal_dependent_features(
    marginal_pvals: np.ndarray,
    selected_columns: np.ndarray,
    alpha: float,
) -> List[int]:
    """Get features that exhibit marginal dependence with the target variable."""
    # Select p-values corresponding to the candidate columns
    subset_pvals = marginal_pvals[selected_columns]

    # Apply Benjamini–Hochberg FDR
    rejected = _benjamini_hochberg(subset_pvals, alpha)

    # Map rejected subset indices back to original feature indices
    selected = selected_columns[rejected]

    return selected.astype(int).tolist()


def _test_conditional_independence(
    x: np.ndarray,
    y: np.ndarray,
    j: int,
    others: List[int],
    alpha: float,
    max_cond_set_size: int = 3,
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
    :param alpha: Significance level for partial distance
                  correlation tests.
    :param max_cond_set_size: Maximum number of conditional sets to consider.

    :return: True if ``X_j`` is conditionally independent of ``Y``
             given some subset of others; otherwise False.
    """
    if len(others) == 0:
        return False

    # If the conditioning set is too large, randomly sample a subset
    if len(others) > max_cond_set_size:
        rng = np.random.default_rng(seed=j)  # reproducible
        others_subset = rng.choice(others, size=max_cond_set_size, replace=False)
    else:
        others_subset = others

    p = partial_dcor(x[:, j], y, cond=x[:, others_subset])
    return p > alpha


def _remove_conditional_independent_features(
    x: np.ndarray,
    y: np.ndarray,
    selected: List[int],
    alpha: float,
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
    :param alpha: Significance level for partial distance
                  correlation tests.
    :return: Sorted list of feature indices that remain after
             iterative pruning.
    """
    selected_set = set(selected)
    changed = True

    while changed:
        changed = False
        for j in list(selected_set):
            others = list(selected_set - {j})
            if _test_conditional_independence(
                x=x,
                y=y,
                j=j,
                others=others,
                alpha=alpha,
            ):
                selected_set.remove(j)
                changed = True

    return list(selected_set)


def _update_resample_schedule(
    step: int,
    n_resamples: int,
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
    period = max(1, int(n_resamples * 0.1))
    if step % period != 0:
        num_features_to_select = int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0
        return feature_count, non_zero_columns, num_features_to_select

    non_zero_columns = np.flatnonzero(feature_count).tolist()
    num_features_to_select = int(np.sqrt(len(non_zero_columns))) if non_zero_columns else 0

    if step != n_resamples:
        feature_count.fill(0)

    return feature_count, non_zero_columns, num_features_to_select


def _final_selection(feature_count: np.ndarray) -> List[int]:
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
    non_zero_idx = np.flatnonzero(feature_count)
    if non_zero_idx.size == 0:
        return []

    non_zero_values = feature_count[non_zero_idx]
    std = float(np.std(non_zero_values))
    threshold_diff = 2.0 * std

    if np.min(non_zero_values) > threshold_diff:
        return non_zero_idx.tolist()

    median_non_zero = np.median(non_zero_values)
    return non_zero_idx[non_zero_values > median_non_zero].tolist()
