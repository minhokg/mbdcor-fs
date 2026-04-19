import dcor
import numpy as np
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge


def partial_dcor(x: np.ndarray, y: np.ndarray, cond: np.ndarray = None, alpha: float = 0.01) -> float:
    """
    Compute a p-value for testing (conditional) independence using distance correlation.

    :param x: Predictor variable. Array of shape (n,) or (n,1).
    :param y: Target variable (categorical or continuous). Array of shape (n,) or (n,1).
    :param cond: Conditioning matrix Z of shape (n,k), or None. If provided, tests whether x ⟂ y | Z.
    :param alpha: Regularization parameter (Ridge alpha, Logistic uses C=1/alpha).

    :return: Two-sided p-value for the null hypothesis that the (partial)
             distance correlation between x and y is zero.

    :note:
        - The function uses Ridge regression for continuous variables and
          Logistic regression for categorical variables to compute residuals.
        - Distance correlation is calculated for residuals of `x` and `y` after
          adjusting for the conditioning variables (if provided).
        - The function returns a two-sided p-value based on the distance correlation
          and a t-statistic approximation using degrees of freedom.
    """
    # Convert x and y to arrays and reshape to 2D
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    assert x.shape[0] == y.shape[0], "x and y must have the same number of samples"
    n = x.shape[0]

    # Detect whether variables are categorical (integer) or continuous
    x_is_class = np.issubdtype(x.dtype, np.integer)
    y_is_class = np.issubdtype(y.dtype, np.integer)
    if y_is_class:
        y = y.astype(float)

    if cond is None:
        rx = x
        ry = y
    else:
        z = np.asarray(cond)
        assert z.shape[0] == n, "cond and x must have the same number of samples"

        if z.shape[1] > 5:
            pca = PCA(n_components=min(5, z.shape[1]))
            z = pca.fit_transform(z)

        # Add intercept
        z1 = np.column_stack([np.ones(n), z])

        # ----- Residuals for y ~ Z -----
        if y_is_class:
            log_reg_y = LogisticRegression(C=1 / alpha, l1_ratio=0, solver="lbfgs")
            log_reg_y.fit(z1, y.ravel())
            prob_y = log_reg_y.predict_proba(z1)[:, 1].reshape(-1, 1)
            ry = y - prob_y
        else:
            ridge_y = Ridge(alpha=alpha)
            ridge_y.fit(z1, y.ravel())
            ry = y - ridge_y.predict(z1).reshape(-1, 1)

        # ----- Residuals for x ~ Z -----
        if x_is_class:
            log_reg_x = LogisticRegression(C=1 / alpha, l1_ratio=0, solver="lbfgs")
            log_reg_x.fit(z1, x.ravel())
            prob_x = log_reg_x.predict_proba(z1)[:, 1].reshape(-1, 1)
            rx = x - prob_x
        else:
            ridge_x = Ridge(alpha=alpha)
            ridge_x.fit(z1, x.ravel())
            rx = x - ridge_x.predict(z1).reshape(-1, 1)

        # Distance correlation of residuals
    dcor_value = dcor.u_distance_correlation_sqr(
        x=rx,
        y=ry,
        method="mergesort",
    )

    # Degrees of freedom approximation
    df = n * (n - 3) / 2 - 1

    # t-statistic
    t_stat = np.sqrt(df) * dcor_value / np.sqrt(1 - dcor_value**2)

    # Two-sided p-value
    p_value = t.sf(np.abs(t_stat), df=df) * 2

    return float(p_value)
