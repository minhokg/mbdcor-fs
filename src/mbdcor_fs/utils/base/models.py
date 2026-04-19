from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RunResult:
    """
    Container for summarizing the outcome of a single simulation or model run.

    :param method: Name or identifier of the method/algorithm used.
    :param n: Sample size used in the run.
    :param p: Number of features, parameters, or variables.
    :param sim: Simulation or repetition index.
    :param runtime_s: Execution time in seconds.
    :param recall_pct: Recall expressed as a percentage (0–100).
    :param n_selected: Number of selected features/items.
    :param mse: Mean Squared Error, if applicable.
    :param log_loss: Logarithmic loss, if applicable.
    """

    method: str
    n: int
    p: int
    sim: int
    runtime_s: float
    recall_pct: float
    n_selected: int
    mse: Optional[float] = None
    log_loss: Optional[float] = None


@dataclass
class LocalPermutationResult:
    """
    Result of a local permutation test.

    :param p_value: Estimated p-value from the permutation test.
    :param t_obs: Observed test statistic.
    :param t_perm: Array of test statistics computed from permutations.
    :param reject: Whether the null hypothesis is rejected at the chosen significance level.
    :param bins: Bin assignments or grouping used during the permutation procedure.
    """

    p_value: float
    t_obs: float
    t_perm: np.ndarray
    reject: bool
    bins: np.ndarray
