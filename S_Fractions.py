import numpy as np
from S_Symbolic import (
    calculate_beta_p,
    calculate_beta_r,
    calculate_beta_n,
    calculate_delta_p,
    calculate_delta_r,
    calculate_omega,
    calculate_gamma,
    calculate_objective_min,
    numerical_jac_objective_value,
    numerical_hess_objective_value
)


def calculate_objective(F, c, d, sigma, vMax, X):
    """
    Calculate objective for minimization
    Args: (F, c, d, sigma, vMax, X) where F, c, d are decision variables
    Returns: Negative profit value (for minimization)
    """
    return calculate_objective_min(F, c, d, sigma, vMax, X)


def calculate_gamma_omega(F, c, d, sigma, vMax, X):
    """
    Calculate gamma and omega values
    Args: (F, c, d, sigma, vMax, X) where F, c, d are decision variables
    Returns: tuple (gamma, omega)
    """
    gamma = calculate_gamma(F, c, d, sigma, vMax, X)
    omega = calculate_omega(F, c, d, sigma, vMax, X)
    return gamma, omega


def calculate_all_variables(F, c, d, sigma, vMax, X):
    """
    Calculate all beta and delta variables at once
    Args: (F, c, d, sigma, vMax, X) where F, c, d are decision variables
    Returns: tuple (beta_p, beta_r, beta_n, delta_p, delta_r)
    """
    beta_p = calculate_beta_p(F, c, d, sigma, vMax, X)
    beta_r = calculate_beta_r(F, c, d, sigma, vMax, X)
    beta_n = calculate_beta_n(F, c, d, sigma, vMax, X)
    delta_p = calculate_delta_p(F, c, d, sigma, vMax, X)
    delta_r = calculate_delta_r(F, c, d, sigma, vMax, X)

    return beta_p, beta_r, beta_n, delta_p, delta_r


def verify_beta_sum(F, c, d, sigma, vMax, X):
    """
    Verify that β^P + β^R + β^N = 1 (should always be true)
    Args: (F, c, d, sigma, vMax, X) where F, c, d are decision variables
    """
    beta_p = calculate_beta_p(F, c, d, sigma, vMax, X)
    beta_r = calculate_beta_r(F, c, d, sigma, vMax, X)
    beta_n = calculate_beta_n(F, c, d, sigma, vMax, X)

    total = beta_p + beta_r + beta_n
    print(f"β^P + β^R + β^N = {total:.6f} (should be 1.0)")
    return total


def calculate_all_variables_array(x, sigma, vMax, X):
    """
    Calculate all variables with array input
    Args: x = [F, c, d], followed by (sigma, vMax, X)
    """
    return calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)


def get_actual_profit(F, c, d, sigma, vMax, X):
    """
    Get the actual profit value (positive)
    Args: (F, c, d, sigma, vMax, X) where F, c, d are decision variables
    Returns: Actual profit (positive value)
    """
    min_obj = calculate_objective(F, c, d, sigma, vMax, X)
    return min_obj  # Negate to get actual profit


# ============================================================================
# ARRAY WRAPPERS FOR COMPATIBILITY
# ============================================================================

def calculate_objective_array(x, sigma, vMax, X):
    """
    Wrapper for array input
    Args: x = [F, c, d], followed by (sigma, vMax, X)
    """
    return calculate_objective(x[0], x[1], x[2], sigma, vMax, X)


# ============================================================================
# OPTIMIZATION HELPER FUNCTIONS
# ============================================================================

def display_optimization_result(result, sigma, vMax, X):
    """
    Display optimization results with correct profit value
    """
    if result.success:
        F_opt, c_opt, d_opt = result.x
        min_objective = result.fun
        actual_profit = -min_objective  # Convert back to maximization value

        print(f"\n✅ OPTIMIZATION SUCCESSFUL!")
        print(f"   F* = {F_opt:.6f}")
        print(f"   c* = {c_opt:.6f}")
        print(f"   d* = {d_opt:.6f}")
        print(f"   Profit* = {actual_profit:.6f}")
        print(f"   (Minimization objective = {min_objective:.6f})")

        # Also display the components
        beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(
            F_opt, c_opt, d_opt, sigma, vMax, X
        )

        print(f"\n   Component values at optimum:")
        print(f"   β^P = {beta_p:.6f}")
        print(f"   β^R = {beta_r:.6f}")
        print(f"   β^N = {beta_n:.6f}")
        print(f"   δ^P = {delta_p:.6f}")
        print(f"   δ^R = {delta_r:.6f}")

    else:
        print(f"\n❌ OPTIMIZATION FAILED: {result.message}")


def check_gradient_consistency(F, c, d, sigma, vMax, X, epsilon=1e-6):
    """
    Verify that analytical and numerical gradients match
    """
    from scipy.optimize import approx_fprime

    x = np.array([F, c, d])
    args = (sigma, vMax, X)

    # Analytical gradient
    grad_analytical = np.array(numerical_jac_objective_value(F, c, d, sigma, vMax, X)).flatten()

    # Numerical gradient
    grad_numerical = approx_fprime(x, calculate_objective_array, epsilon, *args)

    # Compare
    diff = np.linalg.norm(grad_analytical - grad_numerical)
    print(f"Gradient consistency check:")
    print(f"  Analytical: {grad_analytical}")
    print(f"  Numerical:  {grad_numerical}")
    print(f"  Difference: {diff:.2e}")

    return diff < 1e-4


# ============================================================================
# JACOBIAN AND HESSIAN WRAPPERS FOR OPTIMIZATION
# ============================================================================

def jac_obj(x, *args):
    """
    Jacobian of objective function for minimization

    Since we're minimizing -π, the gradient is already correct from symbolic
    """
    sigma, vMax, X = args
    F = x[0]
    c = x[1]
    d = x[2]

    try:
        # Get gradient from symbolic calculation
        jac_array = np.array(numerical_jac_objective_value(F, c, d, sigma, vMax, X)).flatten()
        return jac_array

    except Exception as e:
        print(f"Warning: Using numerical gradient due to: {e}")
        # Fallback to numerical gradient
        from scipy.optimize import approx_fprime
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, calculate_objective_array, epsilon, *args)


def hess_obj(x, *args):
    """
    Hessian of objective function for minimization

    Returns LinearOperator for scipy optimization
    """
    from scipy.sparse.linalg import LinearOperator

    sigma, vMax, X = args
    F = x[0]
    c = x[1]
    d = x[2]

    def matvec(v):
        try:
            # Get Hessian from symbolic calculation
            H = numerical_hess_objective_value(F, c, d, sigma, vMax, X)
            return np.dot(H, v)

        except Exception as e:
            print(f"Warning: Hessian computation failed: {e}")
            # Fallback to identity (conservative)
            return v

    return LinearOperator((len(x), len(x)), matvec=matvec)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_solution_sensitivity(F_opt, c_opt, d_opt, sigma, vMax, X, delta=0.01):
    """
    Analyze sensitivity of the solution to small perturbations
    """
    print("\n=== Sensitivity Analysis ===")

    # Get baseline values
    baseline_profit = get_actual_profit(F_opt, c_opt, d_opt, sigma, vMax, X)
    baseline_betas = calculate_all_variables(F_opt, c_opt, d_opt, sigma, vMax, X)

    # Test sensitivity to each variable
    variables = [
        ('F', F_opt * (1 + delta), c_opt, d_opt),
        ('c', F_opt, c_opt * (1 + delta), d_opt),
        ('d', F_opt, c_opt, min(0.99, d_opt * (1 + delta)))
    ]

    for var_name, F_test, c_test, d_test in variables:
        new_profit = get_actual_profit(F_test, c_test, d_test, sigma, vMax, X)
        profit_change = (new_profit - baseline_profit) / baseline_profit * 100

        print(f"\n{var_name} increased by {delta * 100:.1f}%:")
        print(f"  Profit change: {profit_change:.4f}%")

        new_betas = calculate_all_variables(F_test, c_test, d_test, sigma, vMax, X)
        print(f"  β^P change: {(new_betas[0] - baseline_betas[0]):.6f}")
        print(f"  β^R change: {(new_betas[1] - baseline_betas[1]):.6f}")
        print(f"  β^N change: {(new_betas[2] - baseline_betas[2]):.6f}")


def check_optimality_conditions(F_opt, c_opt, d_opt, sigma, vMax, X, tol=1e-6):
    """
    Check first-order optimality conditions
    """
    print("\n=== Optimality Conditions Check ===")

    # Check gradient (should be close to zero at optimum)
    grad = numerical_jac_objective_value(F_opt, c_opt, d_opt, sigma, vMax, X)
    grad_norm = np.linalg.norm(grad)

    print(f"Gradient at optimum: {np.array(grad).flatten()}")
    print(f"Gradient norm: {grad_norm:.2e}")
    print(f"First-order condition satisfied: {'✓' if grad_norm < tol else '✗'}")

    # Check Hessian (should be positive semi-definite for minimization)
    H = numerical_hess_objective_value(F_opt, c_opt, d_opt, sigma, vMax, X)
    eigenvalues = np.linalg.eigvals(H)

    print(f"\nHessian eigenvalues: {eigenvalues}")
    print(f"Minimum eigenvalue: {np.min(eigenvalues):.6f}")
    print(f"Second-order condition satisfied: {'✓' if np.all(eigenvalues >= -tol) else '✗'}")

    return grad_norm < tol and np.all(eigenvalues >= -tol)


# # ============================================================================
# # TESTING AND VALIDATION
# # ============================================================================
#
# if __name__ == "__main__":
#     # Test parameters
#     F_test = 5.0
#     c_test = 2.0
#     d_test = 0.5  # d is now a decision variable
#     sigma_test = 1.5
#     vMax_test = 10.0
#     X_test = 100.0
#
#     print("=== Testing Optimization Utilities (3 Variables) ===\n")
#
#     # Test basic calculations
#     print(f"Test parameters: F={F_test}, c={c_test}, d={d_test}")
#     print(f"sigma={sigma_test}, vMax={vMax_test}, X={X_test}\n")
#
#     # Calculate all variables
#     beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(
#         F_test, c_test, d_test, sigma_test, vMax_test, X_test
#     )
#
#     print(f"β^P = {beta_p:.6f}")
#     print(f"β^R = {beta_r:.6f}")
#     print(f"β^N = {beta_n:.6f}")
#     print(f"δ^P = {delta_p:.6f}")
#     print(f"δ^R = {delta_r:.6f}")
#
#     # Test objective
#     min_obj = calculate_objective(F_test, c_test, d_test, sigma_test, vMax_test, X_test)
#     actual_profit = get_actual_profit(F_test, c_test, d_test, sigma_test, vMax_test, X_test)
#
#     print(f"\nObjective for minimization: {min_obj:.6f}")
#     print(f"Actual profit: {actual_profit:.6f}")
#
#     # Verify beta sum
#     print("\nVerification:")
#     verify_beta_sum(F_test, c_test, d_test, sigma_test, vMax_test, X_test)
#
#     # Check gradient consistency
#     print(f"\n{'-' * 50}")
#     check_gradient_consistency(F_test, c_test, d_test, sigma_test, vMax_test, X_test)
#
#     # Test gamma and omega
#     gamma, omega = calculate_gamma_omega(F_test, c_test, d_test, sigma_test, vMax_test, X_test)
#     print(f"\nGamma = {gamma:.6f}")
#     print(f"Omega = {omega:.6f}")
#
#     # Test array wrapper
#     print(f"\n{'-' * 50}")
#     print("Testing array wrapper functions:")
#     x_array = np.array([F_test, c_test, d_test])
#     obj_array = calculate_objective_array(x_array, sigma_test, vMax_test, X_test)
#     vars_array = calculate_all_variables_array(x_array, sigma_test, vMax_test, X_test)
#     print(f"Objective from array: {obj_array:.6f}")
#     print(f"Variables from array: β^P={vars_array[0]:.6f}, β^R={vars_array[1]:.6f}, β^N={vars_array[2]:.6f}")