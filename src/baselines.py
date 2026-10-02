"""Machine-learning models used for comparison with the Bayesian Network."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from . import config


def baseline_models() -> dict:
    """Return the comparison models with the settings used in the paper."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE),
    }
