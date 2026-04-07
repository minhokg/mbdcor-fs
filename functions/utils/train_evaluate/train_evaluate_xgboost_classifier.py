import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier


def train_evaluate_xgboost_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> float:
    """
    Train an XGBoost classifier on the training set and evaluate log loss on the test set.

    :param x_train: Training feature matrix of shape (n_train_samples, n_features).
    :param y_train: Training target vector of shape (n_train_samples,).
    :param x_test: Test feature matrix of shape (n_test_samples, n_features).
    :param y_test: Test target vector of shape (n_test_samples,).
    :param random_state: Random seed for reproducibility.
    :return: Log loss on the test set.
    """
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Ensure feature matrices are 2D
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
    if x_test.ndim == 1:
        x_test = x_test.reshape(-1, 1)

    # Basic validation
    if x_train.shape[1] == 0:
        raise ValueError("x_train has no features. Cannot train XGBoost.")
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError("x_test must have the same number of features as x_train.")
    if x_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Training data is empty.")
    if x_test.shape[0] == 0 or y_test.shape[0] == 0:
        raise ValueError("Test data is empty.")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have the same number of samples.")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("x_test and y_test must have the same number of samples.")

    # Initialize model
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
    )

    # Fit on training data only
    model.fit(x_train, y_train)

    # Predict probabilities on test data
    y_pred_prob = model.predict_proba(x_test)

    # Compute log loss on test data
    return float(log_loss(y_test, y_pred_prob, labels=model.classes_))
