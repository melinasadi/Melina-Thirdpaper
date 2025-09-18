# sigma = 1.3
X = 10
vMax = 1
ub_f = 400

lb = [0.005] + [0.005]
ub = [ub_f] + [100]
bound = (lb, ub)


# parameters.py  (your main parameter file)

# Base parameters
params = {
    "sigma": 1.3,
    "X": 10.0,
    "vMax": 1,
    "ub_f": 400.0,
    "lb": [0.005, 0.005],
    "ub": [400.0, 100.0],
}

def update_parameters(base_params, **new_values):
    updated = base_params.copy()
    updated.update(new_values)
    return updated

