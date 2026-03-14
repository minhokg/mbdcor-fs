import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def train_evaluate_xgboost_classifier(x: np.ndarray, y: np.ndarray) -> float:
    """
    Train an XGBoost classifier on input features and evaluate its performance using log loss.

    :param x: Input feature matrix of shape (n_samples, n_features). Can be 1D or 2D.
    :param y: Target labels of shape (n_samples,). Should be categorical (binary or multiclass).
    :return: Log loss of the classifier on the test set.
    """
    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Check if there are any features
    if x.shape[1] == 0:
        raise ValueError("Input x has no features. Cannot train XGBoost.")

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Check that training data is not empty
    if x_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Training data is empty after train-test split.")

    # Initialize the XGBClassifier
    model = XGBClassifier(eval_metric="logloss")

    # Fit the model
    model.fit(x_train, y_train)

    # Predict probabilities on the test set
    y_pred_prob = model.predict_proba(x_test)

    # Calculate log loss
    log_loss_value = log_loss(y_test, y_pred_prob, labels=model.classes_)

    # Return the log loss
    return log_loss_value
