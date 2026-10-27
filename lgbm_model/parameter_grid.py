"""Initial LightGBM hyperparameter configurations for model validation."""


COMMON_PARAMETERS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
    "verbosity": -1,
}


def _configuration(**overrides):
    """Return one complete configuration with reproducible defaults."""
    return {**COMMON_PARAMETERS, **overrides}


INITIAL_PARAMETER_GRID = [
    _configuration(
        learning_rate=0.05,
        num_iterations=500,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        bagging_fraction=1.0,
        bagging_freq=0,
        feature_fraction=1.0,
    ),
    _configuration(
        learning_rate=0.05,
        num_iterations=500,
        num_leaves=15,
        max_depth=6,
        min_child_samples=50,
        bagging_fraction=1.0,
        bagging_freq=0,
        feature_fraction=0.8,
    ),
    _configuration(
        learning_rate=0.05,
        num_iterations=750,
        num_leaves=31,
        max_depth=10,
        min_child_samples=50,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
    ),
    _configuration(
        learning_rate=0.03,
        num_iterations=1000,
        num_leaves=63,
        max_depth=12,
        min_child_samples=50,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
    ),
    _configuration(
        learning_rate=0.03,
        num_iterations=1000,
        num_leaves=127,
        max_depth=14,
        min_child_samples=100,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.7,
    ),
    _configuration(
        learning_rate=0.10,
        num_iterations=300,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
    ),
    _configuration(
        learning_rate=0.02,
        num_iterations=1500,
        num_leaves=31,
        max_depth=10,
        min_child_samples=100,
        bagging_fraction=0.9,
        bagging_freq=1,
        feature_fraction=0.9,
    ),
    _configuration(
        learning_rate=0.05,
        num_iterations=750,
        num_leaves=31,
        max_depth=10,
        min_child_samples=100,
        bagging_fraction=0.8,
        bagging_freq=1,
        feature_fraction=0.8,
        lambda_l1=0.1,
        lambda_l2=1.0,
    ),
    _configuration(
        learning_rate=0.05,
        num_iterations=750,
        num_leaves=63,
        max_depth=12,
        min_child_samples=50,
        bagging_fraction=0.9,
        bagging_freq=1,
        feature_fraction=0.6,
    ),
    _configuration(
        learning_rate=0.03,
        num_iterations=1000,
        num_leaves=63,
        max_depth=12,
        min_child_samples=100,
        bagging_fraction=0.7,
        bagging_freq=1,
        feature_fraction=0.7,
    ),
]
