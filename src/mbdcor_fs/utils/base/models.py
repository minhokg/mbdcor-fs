from dataclasses import dataclass
from typing import Optional


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
    :param precision:
    :param mse: Mean Squared Error, if applicable.
    :param log_loss: Logarithmic loss, if applicable.
    """

    method: str
    n: int
    p: int
    sim: int
    runtime_s: float
    recall_pct: float
    precision_pct: float
    f1_score: float
    n_selected: int
    mse: Optional[float] = None
    log_loss: Optional[float] = None
