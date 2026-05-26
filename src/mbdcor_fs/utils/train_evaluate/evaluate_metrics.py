import numpy as np


def recall_percent(selected: list, truth: np.ndarray) -> float:
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


def precision_percent(selected: list, truth: np.ndarray) -> float:
    """
    Compute precision of selected features.

    :param selected: Selected feature indices.
    :param truth: Ground truth feature indices.
    :return: Precision expressed as a percentage.
    """
    sel = np.array(list(selected))
    if len(sel) == 0:
        return 0.0
    return 100.0 * np.intersect1d(sel, truth).size / len(sel)


def f1_score(selected: list, truth: np.ndarray) -> float:
    """
    Compute F1 score of selected features.

    :param selected: Selected feature indices.
    :param truth: Ground truth feature indices.
    :return: F1 score expressed as a percentage.
    """
    recall = recall_percent(selected=selected, truth=truth) / 100.0
    precision = precision_percent(selected=selected, truth=truth) / 100.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
