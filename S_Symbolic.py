import sympy as sp
from sympy import symbols, lambdify, log

F, c, d, sigma, vMax, X = sp.symbols('F c d sigma vMax X', positive=True)
variables = (F, c, d)
parameters = (sigma, vMax, X)


def symbolic_gamma(F, c, d, sigma, vMax, X):
    return (c * d) + ((sigma - 1) * vMax)


def symbolic_omega(F, c, d, sigma, vMax, X):
    return (d + sigma - 1)


def symbolic_beta_p(F, c, d, sigma, vMax, X):
    gamma = symbolic_gamma(F, c, d, sigma, vMax, X)
    omega = symbolic_omega(F, c, d, sigma, vMax, X)

    term1 = ((X * c * (1 - d) + F) / (sigma))
    term2 = (F / sigma) * (log((X * c * omega) / F))
    term3 = (F / (sigma - 1)) * (log(gamma / (c * omega)))

    beta_p = 1 - (1 / (X * vMax)) * (term1 + term2 + term3)
    return beta_p


def symbolic_beta_r(F, c, d, sigma, vMax, X):
    gamma = symbolic_gamma(F, c, d, sigma, vMax, X)
    omega = symbolic_omega(F, c, d, sigma, vMax, X)

    beta_r = ((F) / ((sigma - 1) * X * vMax)) * (log(gamma / (c * omega)))
    return beta_r


def symbolic_beta_n(F, c, d, sigma, vMax, X):
    omega = symbolic_omega(F, c, d, sigma, vMax, X)

    term1_beta_n = ((X * (1 - d) * c) + F)
    term2_beta_n = F * (log((X * c * omega) / (F)))

    beta_n = (1 / (sigma * X * vMax)) * (term1_beta_n + term2_beta_n)
    return beta_n


def symbolic_delta_p(F, c, d, sigma, vMax, X):
    gamma = symbolic_gamma(F, c, d, sigma, vMax, X)
    omega = symbolic_omega(F, c, d, sigma, vMax, X)

    term1 = (X ** 2 * vMax)
    term2 = -((X ** 2 * c * (d - 1)) + (2 * F * X)) / (sigma)
    term3 = (F ** 2) / (sigma * c * omega)
    term4 = -((F ** 2 * (vMax - c)) / (c * omega * gamma))
    term5 = -((X * c * (1 - d)) / (sigma * vMax))

    delta_p = (1 / (2 * X * vMax)) * (term1 + term2 + term3 + term4) + term5
    return delta_p


def symbolic_delta_r(F, c, d, sigma, vMax, X):
    gamma = symbolic_gamma(F, c, d, sigma, vMax, X)
    omega = symbolic_omega(F, c, d, sigma, vMax, X)

    delta_r = (F ** 2 / (2 * X * vMax)) * ((vMax - c) / (c * omega * gamma))
    return delta_r


def symbolic_objective_for_minimization(F, c, d, sigma, vMax, X):

    beta_p = symbolic_beta_p(F, c, d, sigma, vMax, X)
    delta_p = symbolic_delta_p(F, c, d, sigma, vMax, X)
    delta_r = symbolic_delta_r(F, c, d, sigma, vMax, X)

    # Original profit (to be maximized)
    profit = (beta_p * F) + ((1 - d) * c * delta_p) + (c * delta_r)

    # Return negative for minimization
    return -profit


# ============================================================================
# CREATE SYMBOLIC EXPRESSIONS
# ============================================================================

# Individual component expressions
beta_p_expr = symbolic_beta_p(F, c, d, sigma, vMax, X)
beta_r_expr = symbolic_beta_r(F, c, d, sigma, vMax, X)
beta_n_expr = symbolic_beta_n(F, c, d, sigma, vMax, X)
delta_p_expr = symbolic_delta_p(F, c, d, sigma, vMax, X)
delta_r_expr = symbolic_delta_r(F, c, d, sigma, vMax, X)
gamma_expr = symbolic_gamma(F, c, d, sigma, vMax, X)
omega_expr = symbolic_omega(F, c, d, sigma, vMax, X)

# Objective for minimization
objective_min_expr = symbolic_objective_for_minimization(F, c, d, sigma, vMax, X)

# ============================================================================
# COMPUTE SYMBOLIC DERIVATIVES
# ============================================================================

# Jacobians (gradients) with respect to decision variables
jacobian_objective_min = sp.Matrix([objective_min_expr]).jacobian(variables)

# Hessians - 3x3 matrices
hessian_objective_min = sp.hessian(objective_min_expr, variables)

# Component derivatives (if needed)
jacobian_beta_p = sp.Matrix([beta_p_expr]).jacobian(variables)
jacobian_beta_r = sp.Matrix([beta_r_expr]).jacobian(variables)
jacobian_beta_n = sp.Matrix([beta_n_expr]).jacobian(variables)
jacobian_delta_p = sp.Matrix([delta_p_expr]).jacobian(variables)
jacobian_delta_r = sp.Matrix([delta_r_expr]).jacobian(variables)

hessian_beta_p = sp.hessian(beta_p_expr, variables)
hessian_beta_r = sp.hessian(beta_r_expr, variables)
hessian_beta_n = sp.hessian(beta_n_expr, variables)
hessian_delta_p = sp.hessian(delta_p_expr, variables)
hessian_delta_r = sp.hessian(delta_r_expr, variables)

# ============================================================================
# CONVERT TO NUMERICAL FUNCTIONS
# ============================================================================

# Define argument order for lambdify - matching function signatures
arg_order = (F, c, d, sigma, vMax, X)

# Create numerical functions for the expressions
calculate_beta_p = lambdify(arg_order, beta_p_expr, 'numpy')
calculate_beta_r = lambdify(arg_order, beta_r_expr, 'numpy')
calculate_beta_n = lambdify(arg_order, beta_n_expr, 'numpy')
calculate_delta_p = lambdify(arg_order, delta_p_expr, 'numpy')
calculate_delta_r = lambdify(arg_order, delta_r_expr, 'numpy')
calculate_omega = lambdify(arg_order, omega_expr, 'numpy')
calculate_gamma = lambdify(arg_order, gamma_expr, 'numpy')

# Main objective function for minimization
calculate_objective_min = lambdify(arg_order, objective_min_expr, 'numpy')

# Gradient and Hessian of objective for minimization
numerical_jac_objective_value = lambdify(arg_order, jacobian_objective_min, 'numpy')
numerical_hess_objective_value = lambdify(arg_order, hessian_objective_min, 'numpy')

# Component derivatives
numerical_jac_beta_p = lambdify(arg_order, jacobian_beta_p, 'numpy')
numerical_jac_beta_r = lambdify(arg_order, jacobian_beta_r, 'numpy')
numerical_jac_beta_n = lambdify(arg_order, jacobian_beta_n, 'numpy')
numerical_jac_delta_p = lambdify(arg_order, jacobian_delta_p, 'numpy')
numerical_jac_delta_r = lambdify(arg_order, jacobian_delta_r, 'numpy')

numerical_hess_beta_p = lambdify(arg_order, hessian_beta_p, 'numpy')
numerical_hess_beta_r = lambdify(arg_order, hessian_beta_r, 'numpy')
numerical_hess_beta_n = lambdify(arg_order, hessian_beta_n, 'numpy')
numerical_hess_delta_p = lambdify(arg_order, hessian_delta_p, 'numpy')
numerical_hess_delta_r = lambdify(arg_order, hessian_delta_r, 'numpy')

#
# # ============================================================================
# # HELPER FUNCTIONS FOR OPTIMIZATION
# # ============================================================================
#
# def objective_function_for_scipy(x, sigma_val, vMax_val, X_val):
#     """
#     Wrapper function for scipy optimization
#     x = [F, c, d] - array of decision variables
#     Returns scalar objective value (to minimize)
#     """
#     F_val, c_val, d_val = x
#     return calculate_objective_min(F_val, c_val, d_val, sigma_val, vMax_val, X_val)
#
#
# def gradient_for_scipy(x, sigma_val, vMax_val, X_val):
#     """
#     Wrapper function for scipy optimization gradient
#     x = [F, c, d] - array of decision variables
#     Returns gradient as 1D array
#     """
#     F_val, c_val, d_val = x
#     jac = numerical_jac_objective_value(F_val, c_val, d_val, sigma_val, vMax_val, X_val)
#     return jac.flatten()
#
#
# def hessian_for_scipy(x, sigma_val, vMax_val, X_val):
#     """
#     Wrapper function for scipy optimization Hessian
#     x = [F, c, d] - array of decision variables
#     Returns Hessian as 2D array
#     """
#     F_val, c_val, d_val = x
#     return numerical_hess_objective_value(F_val, c_val, d_val, sigma_val, vMax_val, X_val)
#
#
# # ============================================================================
# # VERIFICATION FUNCTION
# # ============================================================================
#
# def verify_gradient_correctness():
#     """
#     Verify that the gradients are computed correctly
#     """
#     import numpy as np
#     from scipy.optimize import approx_fprime
#
#     # Test point
#     F_test, c_test, d_test = 50.0, 30.0, 0.5
#     sigma_test, vMax_test, X_test = 2.0, 100.0, 10.0
#
#     print("=== GRADIENT VERIFICATION ===")
#     print(f"Test point: F={F_test}, c={c_test}, d={d_test}")
#     print(f"Parameters: sigma={sigma_test}, vMax={vMax_test}, X={X_test}\n")
#
#     # Analytical gradient
#     grad_analytical = numerical_jac_objective_value(
#         F_test, c_test, d_test, sigma_test, vMax_test, X_test
#     ).flatten()
#
#     # Numerical gradient
#     x_array = np.array([F_test, c_test, d_test])
#     grad_numerical = approx_fprime(
#         x_array,
#         objective_function_for_scipy,
#         1e-6,
#         sigma_test, vMax_test, X_test
#     )
#
#     # Compare
#     diff = np.linalg.norm(grad_analytical - grad_numerical)
#     print(f"Analytical gradient: {grad_analytical}")
#     print(f"Numerical gradient:  {grad_numerical}")
#     print(f"Difference norm: {diff:.8f}")
#     print(f"Status: {'✓ PASS' if diff < 1e-4 else '✗ FAIL'}")
#
#     # Element-wise comparison
#     print("\nElement-wise comparison:")
#     for i, (a, n) in enumerate(zip(grad_analytical, grad_numerical)):
#         rel_error = abs(a - n) / max(abs(n), 1e-10) * 100
#         var_name = ['F', 'c', 'd'][i]
#         print(f"  ∂f/∂{var_name}: analytical={a:.6f}, numerical={n:.6f}, rel_error={rel_error:.4f}%")
#
#     return diff < 1e-4


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

