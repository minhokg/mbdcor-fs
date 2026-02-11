import dcor
import numpy as np
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge


def partial_dcor(x: np.ndarray, y: np.ndarray, cond: np.ndarray = None, alpha: float = 0.01, random_state: int = None) -> float:
    """
    Compute a p-value for testing (conditional) independence using distance correlation with linear regression residuals.

    This approach is most appropriate when relationships are both linear and non-linear and variables are roughly Gaussian.

    Args:
        x: Shape (n, ) or (n, 1). Predictor variable.
        y: Shape (n, ) or (n, 1). Target variable.
        cond: Conditioning matrix Z of shape (n, k), or None. If provided, tests whether x ⟂ y | Z.
        alpha: Regularization parameter.
        random_state: Optional random state for reproducibility.

    Returns:
        Two-sided p-value for the null hypothesis that the (partial) distance correlation between x and y is zero.

    """
    # Convert x and y to arrays and reshape to 2D
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(float)

    assert x.shape[0] == y.shape[0], "x and y must have the same shape"

    n = x.shape[0]

    if cond is None:
        # If no conditioning, use distance correlation between x and y
        dcor_value = dcor.u_distance_correlation_sqr(
            x=x,
            y=y,
            method="mergesort",
        )

    else:
        z = np.asarray(cond)
        assert z.shape[0] == x.shape[0], "z and x must have the same shape"

        # Add intercept
        z1 = np.column_stack([np.ones(n), z])

        # Residuals of x ~ z

        ridge_x = Ridge(alpha=alpha)
        ridge_x.fit(z1, x.ravel())
        rx = x - ridge_x.predict(z1).reshape(-1, 1)

        # Residuals of y ~ z
        ridge_y = Ridge(alpha=alpha)
        ridge_y.fit(z1, y.ravel())
        ry = y - ridge_y.predict(z1).reshape(-1, 1)

        # Compute distance correlation between residuals rx and ry
        dcor_value = dcor.u_distance_correlation_sqr(
            x=rx,
            y=ry,
            method="mergesort",
        )

    # Degrees of freedom
    df = n * (n - 3) / 2

    # t-statistic for testing significance
    t_stat = np.sqrt(df - 1) * dcor_value / np.sqrt(1 - dcor_value**2)
    p = 1 - student_t.cdf(t_stat, df=df)

    return float(p)
