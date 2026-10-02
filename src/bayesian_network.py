"""Bayesian Network construction, parameter learning and inference (pgmpy).

Works with both the older pgmpy API (BayesianNetwork, estimator passed as a
class) and the newer one (DiscreteBayesianNetwork, estimator passed as an
instance).
"""

import pandas as pd

from . import config

try:  # pgmpy >= 1.0
    from pgmpy.models import DiscreteBayesianNetwork as _BayesianNetwork
except ImportError:  # pgmpy 0.1.x
    from pgmpy.models import BayesianNetwork as _BayesianNetwork

from pgmpy.inference import VariableElimination

try:  # hide pgmpy progress bars
    from pgmpy import config as _pgmpy_config
    _pgmpy_config.set_show_progress(False)
except (ImportError, AttributeError):
    pass


def build_network(parents, target: str = config.TARGET):
    """Create a network in which every variable in `parents` points to `target`."""
    return _BayesianNetwork([(parent, target) for parent in parents])


def fit_mle(model, data: pd.DataFrame):
    """Estimate the conditional probability tables by Maximum Likelihood Estimation."""
    data = data[list(model.nodes())]
    try:  # pgmpy >= 1.1
        from pgmpy.parameter_estimator import DiscreteMLE
        model.fit(data, estimator=DiscreteMLE())
    except ImportError:
        from pgmpy.estimators import MaximumLikelihoodEstimator
        model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model


def predict(model, X: pd.DataFrame, target: str = config.TARGET) -> pd.Series:
    """Most probable target state for each row of X (exact inference)."""
    features = [node for node in model.nodes() if node != target]
    predictions = model.predict(X[features])
    return predictions[target].astype(int)


def query(model, evidence: dict, target: str = config.TARGET) -> dict:
    """Posterior distribution P(target | evidence) using Variable Elimination."""
    result = VariableElimination(model).query(
        variables=[target], evidence=evidence, show_progress=False
    )
    states = result.state_names[target]
    return {int(state): float(prob) for state, prob in zip(states, result.values)}
