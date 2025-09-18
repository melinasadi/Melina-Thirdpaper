
import pandas as pd
import os
from datetime import datetime
from scipy.optimize import Bounds
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint, approx_fprime, differential_evolution
from scipy.sparse.linalg import LinearOperator
from S_parameters_value import params, update_parameters
from S_trust_constru_T_periods import setup_constraints, run_optimization, params_init, calculate_all_variables

# ========= SAVE DIR =========
today = datetime.today().strftime('%Y-%m-%d')
save_dir = os.path.join("results", "optimization", today)
os.makedirs(save_dir, exist_ok=True)

# ========= UTIL (optional debug like your template) =========
def debug_param_status(param, requested_value, actual_tuple):
    s, X, v = actual_tuple
    print(f"🔍 Debug - Parameter {param}:")
    print(f"   Requested: {requested_value}")
    print(f"   Using (sigma, X, vMax) = ({s}, {X}, {v})")

# Ranges (inclusive) – mirrors your template style
param_ranges = {
    "sigma":  np.arange(start=1.1, stop=2, step=0.1),
    # "X":     np.arange(start=1.1, stop=10, step=0.01),
    # "vMax":  np.arange(start=0.1, stop=2, step=0.01),
}

# ========= MAIN SWEEP (one param at a time) =========
for param, values in param_ranges.items():
    # Per-parameter collectors (keyed by the varied value)
    results = {
        "optimal_f": {}, "optimal_c": {}, "optimal_d": {}, "optimal_pi": {},
        "beta_P": {}, "beta_R": {}, "beta_N": {}, "delta_P": {}, "delta_R": {},
        "success": {}, "nit": {}, "message": {}, "constr_violation": {}
    }

    for current_value in values:
        # Get a modified copy
        new_params = update_parameters(params)

        # Initialize ALL values first from the base parameters
        sigma_val = new_params["sigma"]
        X_val = new_params["X"]
        vMax_val = new_params["vMax"]
        ub_f_val = new_params["ub_f"]

        # Then override ONLY the one being varied
        if param == "sigma":
            sigma_val = float(current_value)
        elif param == "X":
            X_val = float(current_value)
        elif param == "vMax":
            vMax_val = float(current_value)

        debug_param_status(param, current_value, (sigma_val, X_val, vMax_val))

        # Now all three values are defined
        bounds = Bounds([0.0, 0.0, 0.0], [ub_f_val, vMax_val, 1.0])
        # Constraints for this parameter combo
        constraints = setup_constraints(sigma_val, vMax_val, X_val)

        # Initial guess (your function uses feasibility search -> DE fallback)
        try:
            x0 = params_init(sigma_val, vMax_val, X_val, ub_f_val)
        except Exception as e:
            print(f"⚠️ params_init failed ({e}); fallback to mid-bounds.")
            x0 = np.array([min(1.0, ub_f_val), min(0.5 * vMax_val + 1e-3, vMax_val), 0.5])

        print(f"🔧 Running: {param} = {current_value} | init = {x0}")

        # Run local solver
        res = run_optimization(x0, sigma_val, vMax_val, X_val, ub_f_val)

        # Default row values
        f_opt = c_opt = d_opt = np.nan
        obj   = np.nan
        prof  = np.nan
        bP = bR = bN = dP = dR = np.nan
        success = getattr(res, "success", False)
        nit = getattr(res, "nit", np.nan)
        msg = getattr(res, "message", "")
        cviol = getattr(res, "constr_violation", np.nan)

        if res is not None and hasattr(res, "x"):
            f_opt, c_opt, d_opt = res.x
            obj = res.fun
            prof = -obj
            try:
                bP, bR, bN, dP, dR = calculate_all_variables(f_opt, c_opt, d_opt, sigma_val, vMax_val, X_val)
            except Exception as e:
                print(f"⚠️ calculate_all_variables failed at {param}={current_value}: {e}")

        # Store results keyed by the parameter value
        key = float(current_value)
        results["optimal_f"][key]  = f_opt
        results["optimal_c"][key]  = c_opt
        results["optimal_d"][key]  = d_opt
        results["optimal_pi"][key] = prof
        results["beta_P"][key] = bP
        results["beta_R"][key] = bR
        results["beta_N"][key] = bN
        results["delta_P"][key] = dP
        results["delta_R"][key] = dR
        results["success"][key] = success
        results["nit"][key] = nit
        results["message"][key] = str(msg)
        results["constr_violation"][key] = cviol

    # --- build DataFrame exactly like your template style ---
    data_rows = []
    for current_value in values:
        key = float(current_value)
        data_rows.append({
            param: key,
            "optimal_f": results["optimal_f"][key],
            "optimal_c": results["optimal_c"][key],
            "optimal_d": results["optimal_d"][key],
            "optimal_pi": results["optimal_pi"][key],
            "beta_P": results["beta_P"][key],
            "beta_R": results["beta_R"][key],
            "beta_N": results["beta_N"][key],
            "delta_P": results["delta_P"][key],
            "delta_R": results["delta_R"][key],
            "success": results["success"][key],
            "nit": results["nit"][key],
            "constr_violation": results["constr_violation"][key],
            "message": results["message"][key],
        })
    df = pd.DataFrame(data_rows)

    # Save CSV (one file per parameter)
    csv_path = os.path.join(save_dir, f"sensitivity_{param}.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV: {csv_path}")
