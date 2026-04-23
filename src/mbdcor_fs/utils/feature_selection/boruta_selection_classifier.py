from typing import List

import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def boruta_selection_classifier(x_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> List[int]:
    """
    Boruta selection using a Random Forest classifier.

    :param x_train: feature of training set
    :param y_train: label of training set
    :param random_state: random seed
    :return: list of selected features
    """
    # create a random forest classifier
    rf = RandomForestClassifier(random_state=random_state)

    # run boruta feature selection
    boruta = BorutaPy(rf, n_estimators="auto", random_state=random_state)
    boruta.fit(x_train, y_train)

    # get selected features
    selected_features_idx = [index for index, value in enumerate(boruta.support_) if value]

    return selected_features_idx
