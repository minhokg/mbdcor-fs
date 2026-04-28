from typing import List

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_selection_classifier(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> List[int]:
    """
    Boruta selection using a Random Forest classifier.

    Args:
        x (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        random_state (int, optional): Random state. Defaults to 42.

    Returns:
        List[int]:

    """
    # create a random forest classifier
    rf = RandomForestClassifier(max_depth=5, random_state=random_state)

    # run boruta feature selection
    boruta = BorutaPy(rf, n_estimators="auto", random_state=random_state)
    boruta.fit(x, y)

    # get selected features
    selected_features_idx = [index for index, value in enumerate(boruta.support_) if value]

    return selected_features_idx
