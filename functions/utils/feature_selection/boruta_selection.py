from typing import List

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def boruta_selection(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> List[int]:
    """
    Boruta selection using a Random Forest classifier.

    Args:
        x (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        random_state (int, optional): Random state. Defaults to 42.

    Returns:
        List[int]:

    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # create a random forest classifier
    rf = RandomForestClassifier(n_jobs=-1, max_depth=5, random_state=random_state)

    # run boruta feature selection
    boruta = BorutaPy(rf, n_estimators="auto", random_state=random_state)
    boruta.fit(x_train, y_train)

    # get selected features
    selected_features_idx = [index for index, value in enumerate(boruta.support_) if value]

    return selected_features_idx
