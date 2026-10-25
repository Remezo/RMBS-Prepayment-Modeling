from itertools import product
from calculate_error import error_summary

def validated_parameters(train,validate):
    parameters={
    'objective': ['binary'],
    'metric': ['binary_logloss'],
    'boosting_type': ['gbdt'],
    'num_leaves': [31,50]
    }

    keys=parameters.keys()

    for combination in product(*parameters.values()):
        current_params=dict(zip(keys,combination))
        return current_params