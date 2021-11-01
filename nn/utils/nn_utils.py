from itertools import product


def hyperparameter_combinations(tuning_dict):
    keys = tuning_dict.keys()
    params = [tuning_dict[key] for key in keys]
    d_out = [dict(zip(keys, i)) for i in product(*params)]
    return d_out
