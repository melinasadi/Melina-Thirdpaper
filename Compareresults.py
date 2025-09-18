import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint, approx_fprime, \
    differential_evolution
from scipy.sparse.linalg import LinearOperator

# Import parameters (note: d is no longer imported as a fixed parameter)
from Main_parameters_value import X, sigma, vMax, lb, ub, ub_f

# Import your mathematical functions from the updated 3-variable symbolic file
from Main_Symbolic import (
    calculate_objective_min,
    numerical_jac_objective_value,
    numerical_hess_objective_value
)

from Main_Fractions import (
    calculate_all_variables,
    verify_beta_sum,
)

from Main_feasibilityregion import find_max_values_3d


# ============================================================================
# INITIALIZATION
# ============================================================================

def params_init(sigma, vMax, X, ub_f):
    """Initialize parameters with fallback options for 3 variables"""
    try:
        # Try to find initial values from feasibility analysis
        F_init, c_init, d_init, _, _ = find_max_values_3d(
            X, sigma, vMax, ub_f,
            plot=True, tolerance=0.01,
            n_F=20, n_c=20, n_d=15
        )

        if F_init is None or c_init is None or d_init is None:
            raise ValueError("Infeasible result from find_max_values_3d")

        print("✅ Initial values found from feasibility analysis.")
        params_initial = np.array([F_init, c_init, d_init])

    except Exception as e:
        # Fallback to global optimization
        print(f"⚠️ Feasibility analysis failed: {e}")
        print("🔁 Using differential evolution for initial guess...")

        # Set up bounds for differential evolution (3 variables)
        bounds_de = [
            (0.01, ub_f),  # F bounds
            (0.01, vMax),  # c bounds
            (0.01, 0.99)  # d bounds
             ]

        result = differential_evolution(
            func=lambda x: calculate_objective_min(x[0], x[1], x[2], sigma, vMax, X),
            bounds=bounds_de,
            strategy='best1bin',
            maxiter=1000,
            polish=True,
            disp=False
        )

        F_init, c_init, d_init = result.x
        print("✅ Initial values found using differential evolution.")
        params_initial = np.array([F_init, c_init, d_init])

    # Report initial values
    initial_profit = calculate_objective_min(F_init, c_init, d_init, sigma, vMax, X)
    print(f"Initial point: F = {F_init:.6f}, c = {c_init:.6f}, d = {d_init:.6f}")
    print(f"Initial profit: {-initial_profit:.6f}")

    return params_initial


# ============================================================================
# OBJECTIVE AND DERIVATIVES
# ============================================================================

def objective_function(x, sigma, vMax, X):
    """Objective function for minimization with 3 variables"""
    return calculate_objective_min(x[0], x[1], x[2], sigma, vMax, X)


def jac_obj(x, *args):
    """Jacobian of objective function (3D)"""
    sigma, vMax, X = args
    F, c, d = x[0], x[1], x[2]

    try:
        # Get jacobian from symbolic calculation
        jac_val = numerical_jac_objective_value(F, c, d, sigma, vMax, X)

        # Debug: compare with numerical gradient
        epsilon = np.sqrt(np.finfo(float).eps)
        jac_numeric = approx_fprime(x, objective_function, epsilon, *args)
        jac_diff = np.array(jac_val).flatten() - jac_numeric

        if np.linalg.norm(jac_diff) > 1e-3:
            print(f"Warning: Large gradient difference: {np.linalg.norm(jac_diff):.6f}")

        return np.array(jac_val).flatten()
    except Exception as e:
        print(f"Warning: Using numerical gradient due to: {e}")
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, objective_function, epsilon, *args)


def hess_obj(x, *args):
    """Hessian of objective function as LinearOperator (3x3)"""
    sigma, vMax, X = args
    F, c, d = x[0], x[1], x[2]

    def matvec(v):
        try:
            H = numerical_hess_objective_value(F, c, d, sigma, vMax, X)
            return np.dot(H, v)
        except:
            return v  # Fallback to identity

    return LinearOperator((len(x), len(x)), matvec=matvec)


# ============================================================================
# CONSTRAINT FUNCTIONS (Updated for 3 variables)
# ============================================================================

def constraint_func3(x, sigma, vMax, X):
    """Beta_P constraint: 0 <= beta_P <= 1"""
    beta_p, _, _, _, _ = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return beta_p


def constraint_func4(x, sigma, vMax, X):
    """Beta_R constraint: 0 <= beta_R <= 1"""
    _, beta_r, _, _, _ = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return beta_r


def constraint_func5(x, sigma, vMax, X):
    """Beta_N constraint: 0 <= beta_N <= 1"""
    _, _, beta_n, _, _ = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return beta_n


def constraint_func6(x, sigma, vMax, X):
    """Sum of betas should equal 1"""
    beta_p, beta_r, beta_n, _, _ = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return (beta_p + beta_r + beta_n) - 1


def constraint_func7(x, sigma, vMax, X):
    """Delta_P constraint: delta_P >= 0"""
    _, _, _, delta_p, _ = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return delta_p


def constraint_func8(x, sigma, vMax, X):
    """Delta_R constraint: delta_R >= 0"""
    _, _, _, _, delta_r = calculate_all_variables(x[0], x[1], x[2], sigma, vMax, X)
    return delta_r


# ============================================================================
# SETUP CONSTRAINTS (Updated for 3 variables)
# ============================================================================

def setup_constraints(sigma, vMax, X):
    """Set up all constraints for the optimization problem with 3 variables"""
    constraints = []

    # Linear constraints (now for 3 variables: [F, c, d])

    # Constraint 1: 0 <= c <= vMax
    A1 = np.array([[0, 1, 0]])  # [0*F + 1*c + 0*d]
    constraint1 = LinearConstraint(A1, [0], [vMax])
    constraints.append(constraint1)

    # Constraint 2: 0 <= d <= 1
    A2 = np.array([[0, 0, 1]])  # [0*F + 0*c + 1*d]
    constraint2 = LinearConstraint(A2, [0], [1])
    constraints.append(constraint2)

    # Constraint 3: F >= 0
    A3 = np.array([[1, 0, 0]])  # [1*F + 0*c + 0*d]
    constraint3 = LinearConstraint(A3, [0], [np.inf])
    constraints.append(constraint3)

    # Note: The constraint F <= X * (d + sigma - 1) * c is now nonlinear since d is a variable
    # We'll add it as a nonlinear constraint
    def f_upper_bound_constraint(x):
        F, c, d = x[0], x[1], x[2]
        return X * (d + sigma - 1) * c - F  # Should be >= 0

    constraints.append(NonlinearConstraint(
        f_upper_bound_constraint, lb=0, ub=np.inf))

    # Nonlinear constraints
    # Beta constraints (0 <= beta <= 1)
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func3(x, sigma, vMax, X), lb=0, ub=1))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func4(x, sigma, vMax, X), lb=0, ub=1))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func5(x, sigma, vMax, X), lb=0, ub=1))

    # Sum of betas = 1
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func6(x, sigma, vMax, X), lb=1 - 1e-6, ub=1 + 1e-6))

    # Delta constraints (>= 0)
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func7(x, sigma, vMax, X), lb=0, ub=np.inf))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func8(x, sigma, vMax, X), lb=0, ub=np.inf))

    return constraints


# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================

def run_optimization(params_init, sigma, vMax, X, ub_f):
    """Run the optimization with 3 variables"""
    # Set up bounds for 3 variables
    bounds = Bounds([0, 0, 0], [ub_f, vMax, 1])
    constraints = setup_constraints(sigma, vMax, X)
    args = (sigma, vMax, X)

    print("\n=== Starting Optimization (3 Variables) ===")

    try:
        result = minimize(
            fun=objective_function,
            x0=params_init,
            args=args,
            method='trust-constr',
            jac=jac_obj,
            hess=hess_obj,
            bounds=bounds,
            constraints=constraints,
            options={
                'initial_tr_radius': 0.5,
                'gtol': 1e-5,
                'xtol': 1e-5,
                'barrier_tol': 1e-5,
                'maxiter': 20000,
                'verbose': 2
            }
        )

        print(f"\nOptimization completed!")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Number of iterations: {result.nit}")

        return result

    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        return None


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def check_constraints(x, sigma, vMax, X):
    """Check if all constraints are satisfied for 3 variables"""
    print("\n=== Constraint Verification ===")

    F, c, d = x[0], x[1], x[2]

    # Linear constraints
    print("\nLinear constraints:")
    print(f"  c = {c:.6f} (should be in [0, {vMax}]): {'✓' if 0 <= c <= vMax else '✗'}")
    print(f"  d = {d:.6f} (should be in [0, 1]): {'✓' if 0 <= d <= 1 else '✗'}")
    print(f"  F = {F:.6f} (should be >= 0): {'✓' if F >= 0 else '✗'}")

    F_max = X * (d + sigma - 1) * c
    print(f"  F <= {F_max:.6f}: {'✓' if F <= F_max + 1e-6 else '✗'}")

    # Nonlinear constraints
    print("\nNonlinear constraints:")
    beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(F, c, d, sigma, vMax, X)

    print(f"  β^P = {beta_p:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_p <= 1 else '✗'}")
    print(f"  β^R = {beta_r:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_r <= 1 else '✗'}")
    print(f"  β^N = {beta_n:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_n <= 1 else '✗'}")

    beta_sum = beta_p + beta_r + beta_n
    print(f"  Sum of betas = {beta_sum:.8f} (should be 1.0): {'✓' if abs(beta_sum - 1) < 1e-6 else '✗'}")

    print(f"  δ^P = {delta_p:.6f} (should be >= 0): {'✓' if delta_p >= -1e-6 else '✗'}")
    print(f"  δ^R = {delta_r:.6f} (should be >= 0): {'✓' if delta_r >= -1e-6 else '✗'}")


def display_results(result, sigma, vMax, X):
    """Pretty-print optimization results as f, c, d and objective value."""
    if result is None:
        print("\n❌ Optimization returned None.")
        return
    if not result.success:
        print("\n⚠️ Optimization did not converge successfully.")
        print(f"Message: {result.message}")
        # You can still display the best-so-far values:
        if hasattr(result, "x"):
            print("\nBest-so-far variables:")
            print(f"{'Variable':<18}Value")
            print("-" * 32)
            print(f"{'f':<18}{result.x[0]:.6f}")
            print(f"{'c':<18}{result.x[1]:.6f}")
            print(f"{'d':<18}{result.x[2]:.6f}")
            print("-" * 32)
            print(f"{'Objective Value':<18}{result.fun:.6f}")
        return

    f_opt, c_opt, d_opt = result.x
    obj_min = result.fun           # your objective is minimized
    obj_max = -obj_min             # if it's negative-profit min, this is profit

    print("\n=== Optimization Results ===")
    print(f"{'Variable':<18}Value")
    print("-" * 32)
    print(f"{'f':<18}{f_opt:.6f}")
    print(f"{'c':<18}{c_opt:.6f}")
    print(f"{'d':<18}{d_opt:.6f}")
    print("-" * 32)
    print(f"{'Reported Profit π':<18}{obj_max:.6f}")

    # Optional: solver diagnostics
    print("\n--- Solver Summary ---")
    print(f"Success:           {result.success}")
    print(f"Message:           {result.message}")
    print(f"Iterations (nit):  {result.nit}")
    if hasattr(result, "constr_violation"):
        print(f"Max constr viol.:  {result.constr_violation:.3e}")

    # Verify constraints & show derived variables at optimum
    check_constraints(result.x, sigma, vMax, X)

    # Derived variables at optimum (β^P, β^R, β^N, δ^P, δ^R)
    beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(
        f_opt, c_opt, d_opt, sigma, vMax, X
    )
    print("\n--- Derived Variables at Optimum ---")
    print(f"{'β^P':<6}= {beta_p:.6f}")
    print(f"{'β^R':<6}= {beta_r:.6f}")
    print(f"{'β^N':<6}= {beta_n:.6f}")
    print(f"{'δ^P':<6}= {delta_p:.6f}")
    print(f"{'δ^R':<6}= {delta_r:.6f}")

    # Profit verification using your objective function
    verify_profit = -calculate_objective_min(f_opt, c_opt, d_opt, sigma, vMax, X)
    print(f"\nProfit verification: {verify_profit:.6f} (should match {obj_max:.6f})")


# def display_results(result, sigma, vMax, X):
#     """Display optimization results for 3 variables"""
#     if result is None or not result.success:
#         print("\nOptimization failed!")
#         return
#
#     F_opt, c_opt, d_opt = result.x
#     optimal_profit = -result.fun  # Convert back to positive (actual profit)
#
#     print("\n=== Optimization Results (3 Variables) ===")
#     print(f"Optimal F: {F_opt:.6f}")
#     print(f"Optimal c: {c_opt:.6f}")
#     print(f"Optimal d: {d_opt:.6f}")
#     print(f"Optimal profit: {optimal_profit:.6f}")
#
#     # Calculate all variables at optimal point
#     beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(F_opt, c_opt, d_opt, sigma, vMax, X)
#
#     print(f"\nOptimal Variables:")
#     print(f"  β^P = {beta_p:.6f}")
#     print(f"  β^R = {beta_r:.6f}")
#     print(f"  β^N = {beta_n:.6f}")
#     print(f"  δ^P = {delta_p:.6f}")
#     print(f"  δ^R = {delta_r:.6f}")
#
#     # Verify profit calculation
#     verify_profit = -calculate_objective_min(F_opt, c_opt, d_opt, sigma, vMax, X)
#     print(f"\nProfit verification: {verify_profit:.6f} (should match {optimal_profit:.6f})")
#
#     # Check constraints
#     check_constraints(result.x, sigma, vMax, X)


# ============================================================================
# MULTI-START OPTIMIZATION (3 Variables)
# ============================================================================

# def run_multistart_optimization(sigma, vMax, X, ub_f, n_starts=200):
#     """Run optimization from multiple starting points in 3D space"""
#     print(f"\n{'=' * 60}")
#     print(f"MULTI-START OPTIMIZATION ({n_starts} starting points, 3 variables)")
#     print(f"{'=' * 60}")
#
#     # Setup for optimization
#     bounds = Bounds([1e-6, 1e-6, 1e-6], [ub_f, vMax, 1.0])
#     constraints = setup_constraints(sigma, vMax, X)
#     args = (sigma, vMax, X)
#
#     # Generate random starting points in 3D
#     np.random.seed(42)  # For reproducibility
#     F_starts = np.random.uniform(0.01, ub_f * 0.9, n_starts)
#     c_starts = np.random.uniform(0.01, vMax * 0.9, n_starts)
#     d_starts = np.random.uniform(0.01, 0.99, n_starts)
#
#     # Storage for results
#     successful_results = []
#     failed_count = 0
#     boundary_solutions = []
#
#     print("\nRunning optimizations...")
#
#     for i in range(n_starts):
#         if (i + 1) % 20 == 0:
#             print(f"Progress: {i + 1}/{n_starts} completed...")
#
#         x0 = np.array([F_starts[i], c_starts[i], d_starts[i]])
#
#         try:
#             # Run optimization silently
#             result = minimize(
#                 fun=objective_function,
#                 x0=x0,
#                 args=args,
#                 method='trust-constr',
#                 jac=jac_obj,
#                 hess=hess_obj,
#                 bounds=bounds,
#                 constraints=constraints,
#                 options={
#                     'gtol': 1e-10,
#                     'xtol': 1e-10,
#                     'maxiter': 1000,
#                     'verbose': 0  # Silent mode
#                 }
#             )
#
#             if result.success:
#                 # Verify constraints
#                 beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(
#                     result.x[0], result.x[1], result.x[2], sigma, vMax, X
#                 )
#
#                 constraints_satisfied = (
#                         0 <= beta_p <= 1 and
#                         0 <= beta_r <= 1 and
#                         0 <= beta_n <= 1 and
#                         abs(beta_p + beta_r + beta_n - 1) < 1e-6 and
#                         delta_p >= -1e-6 and
#                         delta_r >= -1e-6 and
#                         result.x[0] >= 0 and
#                         result.x[1] >= 0 and
#                         0 <= result.x[2] <= 1
#                 )
#
#                 if constraints_satisfied:
#                     # Check if it's a boundary solution
#                     if beta_r < 1e-6:  # Regular users essentially zero
#                         boundary_solutions.append(result)
#                     else:
#                         successful_results.append(result)
#                 else:
#                     failed_count += 1
#             else:
#                 failed_count += 1
#
#         except Exception as e:
#             failed_count += 1
#
#     print(f"\n{'=' * 60}")
#     print("MULTI-START SUMMARY (3 Variables)")
#     print(f"{'=' * 60}")
#     print(f"Total runs: {n_starts}")
#     print(f"Successful (interior): {len(successful_results)}")
#     print(f"Boundary solutions: {len(boundary_solutions)}")
#     print(f"Failed: {failed_count}")
#
#     # Find best solution
#     all_valid = successful_results + boundary_solutions
#
#     if not all_valid:
#         print("\n❌ No valid solutions found!")
#         return None
#
#     # Find solution with best objective value
#     best_result = min(all_valid, key=lambda r: r.fun)
#     best_is_interior = best_result in successful_results
#
#     # Analyze convergence to best solution
#     best_x = best_result.x
#     tol = 1e-6
#
#     # Count how many converged to best
#     converged_to_best = 0
#     for result in successful_results:
#         if np.linalg.norm(result.x - best_x) < tol:
#             converged_to_best += 1
#
#     # Find unique solutions
#     unique_solutions = []
#     for result in all_valid:
#         is_unique = True
#         for unique_sol, _ in unique_solutions:
#             if np.linalg.norm(result.x - unique_sol) < tol:
#                 is_unique = False
#                 break
#         if is_unique:
#             unique_solutions.append((result.x, -result.fun))
#
#     print(f"\nUnique solutions found: {len(unique_solutions)}")
#
#     if best_is_interior:
#         print(f"\n✅ Best solution (interior):")
#         if len(successful_results) > 0:
#             convergence_rate = converged_to_best / len(successful_results) * 100
#             print(
#                 f"   {converged_to_best}/{len(successful_results)} ({convergence_rate:.1f}%) interior solutions converged here")
#     else:
#         print(f"\n⚠️  Best solution (boundary):")
#
#     print(f"   F* = {best_x[0]:.6f}")
#     print(f"   c* = {best_x[1]:.6f}")
#     print(f"   d* = {best_x[2]:.6f}")
#     print(f"   π* = {-best_result.fun:.6f}")
#
#     # Report alternative solutions if any
#     if len(unique_solutions) > 1:
#         print("\nAlternative solutions:")
#         sorted_solutions = sorted(unique_solutions, key=lambda x: -x[1])  # Sort by profit
#         for i, (sol, profit) in enumerate(sorted_solutions[:3]):  # Show top 3
#             if np.linalg.norm(sol - best_x) > tol:
#                 profit_diff = profit - (-best_result.fun)
#                 beta_p, beta_r, _, _, _ = calculate_all_variables(sol[0], sol[1], sol[2], sigma, vMax, X)
#                 sol_type = "boundary" if beta_r < 1e-6 else "interior"
#                 print(
#                     f"   Solution {i + 1} ({sol_type}): F={sol[0]:.6f}, c={sol[1]:.6f}, d={sol[2]:.6f}, Δπ={profit_diff:.6f}")
#
#     # Report for paper
#     interior_converged_to_best = converged_to_best
#     total_converged = len(successful_results) + len(boundary_solutions)
#
#     print(f"\n📊 For paper reporting:")
#     print(f"   Of {n_starts} runs, {total_converged} ({total_converged / n_starts * 100:.0f}%) converged successfully")
#     if total_converged > 0:
#         print(
#             f"   Among these, {interior_converged_to_best} ({interior_converged_to_best / total_converged * 100:.1f}%) converged to the best interior solution")
#     print(f"   {len(boundary_solutions)} runs terminated at boundary solutions with β^R ≈ 0")
#
#     if len(boundary_solutions) > 0:
#         # Compare best boundary to best interior
#         if best_is_interior and boundary_solutions:
#             best_boundary = min(boundary_solutions, key=lambda r: r.fun)
#             profit_gap = ((-best_result.fun) - (-best_boundary.fun)) / (-best_boundary.fun) * 100
#             print(f"   Best interior solution yields {profit_gap:.1f}% higher profit than best boundary solution")
#
#     return best_result


# ============================================================================
# VISUALIZATION FOR 3D OPTIMIZATION
# ============================================================================

# def visualize_optimization_path(result, sigma, vMax, X, ub_f):
#     """Visualize the optimization landscape and solution"""
#     if result is None:
#         return
#
#     F_opt, c_opt, d_opt = result.x
#
#     fig = plt.figure(figsize=(15, 5))
#
#     # Create grid for visualization
#     n_points = 50
#
#     # Plot 1: F-c slice at optimal d
#     ax1 = fig.add_subplot(131)
#     F_range = np.linspace(0.01, min(ub_f, F_opt * 2), n_points)
#     c_range = np.linspace(0.01, min(vMax, c_opt * 2), n_points)
#     F_grid, c_grid = np.meshgrid(F_range, c_range)
#     Z_Fc = np.zeros_like(F_grid)
#
#     for i in range(n_points):
#         for j in range(n_points):
#             try:
#                 Z_Fc[i, j] = -calculate_objective_min(F_grid[i, j], c_grid[i, j], d_opt, sigma, vMax, X)
#             except:
#                 Z_Fc[i, j] = np.nan
#
#     contour1 = ax1.contour(F_grid, c_grid, Z_Fc, levels=20, cmap='viridis')
#     ax1.clabel(contour1, inline=True, fontsize=8)
#     ax1.scatter(F_opt, c_opt, color='red', s=100, marker='*', zorder=5)
#     ax1.set_xlabel('F')
#     ax1.set_ylabel('c')
#     ax1.set_title(f'Profit contours at d={d_opt:.3f}')
#     ax1.grid(True, alpha=0.3)
#
#     # Plot 2: F-d slice at optimal c
#     ax2 = fig.add_subplot(132)
#     d_range = np.linspace(0.01, 0.99, n_points)
#     F_grid2, d_grid = np.meshgrid(F_range, d_range)
#     Z_Fd = np.zeros_like(F_grid2)
#
#     for i in range(n_points):
#         for j in range(n_points):
#             try:
#                 Z_Fd[i, j] = -calculate_objective_min(F_grid2[i, j], c_opt, d_grid[i, j], sigma, vMax, X)
#             except:
#                 Z_Fd[i, j] = np.nan
#
#     contour2 = ax2.contour(F_grid2, d_grid, Z_Fd, levels=20, cmap='viridis')
#     ax2.clabel(contour2, inline=True, fontsize=8)
#     ax2.scatter(F_opt, d_opt, color='red', s=100, marker='*', zorder=5)
#     ax2.set_xlabel('F')
#     ax2.set_ylabel('d')
#     ax2.set_title(f'Profit contours at c={c_opt:.3f}')
#     ax2.grid(True, alpha=0.3)
#
#     # Plot 3: c-d slice at optimal F
#     ax3 = fig.add_subplot(133)
#     c_grid2, d_grid2 = np.meshgrid(c_range, d_range)
#     Z_cd = np.zeros_like(c_grid2)
#
#     for i in range(n_points):
#         for j in range(n_points):
#             try:
#                 Z_cd[i, j] = -calculate_objective_min(F_opt, c_grid2[i, j], d_grid2[i, j], sigma, vMax, X)
#             except:
#                 Z_cd[i, j] = np.nan
#
#     contour3 = ax3.contour(c_grid2, d_grid2, Z_cd, levels=20, cmap='viridis')
#     ax3.clabel(contour3, inline=True, fontsize=8)
#     ax3.scatter(c_opt, d_opt, color='red', s=100, marker='*', zorder=5)
#     ax3.set_xlabel('c')
#     ax3.set_ylabel('d')
#     ax3.set_title(f'Profit contours at F={F_opt:.3f}')
#     ax3.grid(True, alpha=0.3)
#
#     plt.suptitle(f'Optimization Landscape (Max Profit = {-result.fun:.3f})', fontsize=14)
#     plt.tight_layout()
#     plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================
# ---------- Feasibility & penalty helpers ----------
def is_feasible(x, sigma, vMax, X, tol=1e-6):
    F, c, d = x
    # Linear
    if not (0 <= c <= vMax + tol): return False
    if not (0 <= d <= 1 + tol): return False
    if not (F >= -tol): return False
    # Nonlinear linearized bound on F
    F_max = X * (d + sigma - 1) * c
    if F > F_max + tol: return False
    # Beta/delta constraints
    try:
        beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(F, c, d, sigma, vMax, X)
    except Exception:
        return False
    if not (0 - tol <= beta_p <= 1 + tol): return False
    if not (0 - tol <= beta_r <= 1 + tol): return False
    if not (0 - tol <= beta_n <= 1 + tol): return False
    if abs((beta_p + beta_r + beta_n) - 1) > 1e-6: return False
    if delta_p < -tol or delta_r < -tol: return False
    return True

def objective_safe(x, sigma, vMax, X, big_pen=1e9):
    """Objective with a big penalty if infeasible or error."""
    try:
        if not is_feasible(x, sigma, vMax, X):
            return big_pen
        return calculate_objective_min(x[0], x[1], x[2], sigma, vMax, X)
    except Exception:
        return big_pen
# ---------- Brute-force global (coarse grid) ----------
def brute_force_global(sigma, vMax, X, ub_f,
                       n_F=20, n_c=20, n_d=16,
                       F_min=1e-3, c_min=1e-3, d_min=1e-3, d_max=0.99):
    """
    Coarse grid search to find a good global candidate.
    Only keeps feasible points; returns (x_best, obj_best).
    """
    F_vals = np.linspace(F_min, ub_f, n_F)
    c_vals = np.linspace(c_min, vMax, n_c)
    d_vals = np.linspace(d_min, d_max, n_d)

    best_x, best_obj = None, np.inf
    checked = 0
    kept = 0
    for F in F_vals:
        for c in c_vals:
            for d in d_vals:
                x = np.array([F, c, d], dtype=float)
                checked += 1
                obj = objective_safe(x, sigma, vMax, X)
                if obj < np.inf:
                    kept += 1
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()

    if best_x is None:
        print("❌ Brute force: no feasible point found.")
        return None, None

    print(f"🔎 Brute force checked {checked} grid points; feasible: {kept}.")
    print(f"   Best grid point: f={best_x[0]:.6f}, c={best_x[1]:.6f}, d={best_x[2]:.6f}, objective={best_obj:.6f}")
    return best_x, best_obj
# ---------- Differential Evolution (explicit) ----------
def run_de_global(sigma, vMax, X, ub_f, maxiter=1000):
    bounds_de = [(0.01, ub_f), (0.01, vMax), (0.01, 0.99)]
    # DE on penalized objective to avoid infeasible proposals
    res_de = differential_evolution(
        func=lambda x: objective_safe(x, sigma, vMax, X),
        bounds=bounds_de,
        strategy='best1bin',
        maxiter=maxiter,
        polish=True,
        disp=False
    )
    x_de = res_de.x
    print(f"🧬 DE best: f={x_de[0]:.6f}, c={x_de[1]:.6f}, d={x_de[2]:.6f}, objective={res_de.fun:.6f}")
    return x_de, res_de.fun
# ---------- Compare starts & pick winner ----------
def compare_starts_and_optimize(x0_feas, sigma, vMax, X, ub_f):
    """
    Run local solver from three starts:
      1) feasibility start (x0_feas),
      2) DE global best,
      3) brute-force grid best,
    and print a side-by-side comparison. Returns the best local result.
    """
    # (2) DE candidate
    x_de, _ = run_de_global(sigma, vMax, X, ub_f, maxiter=500)
    # (3) Brute-force candidate
    x_bf, _ = brute_force_global(sigma, vMax, X, ub_f, n_F=24, n_c=24, n_d=18)

    starts = []
    if x0_feas is not None: starts.append(("feasibility", x0_feas))
    if x_de is not None:    starts.append(("DE", x_de))
    if x_bf is not None:    starts.append(("brute-force", x_bf))

    if not starts:
        print("❌ No starting points to compare.")
        return None

    results = []
    for label, x0 in starts:
        print(f"\n▶ Running local solver from [{label}] start: f={x0[0]:.6f}, c={x0[1]:.6f}, d={x0[2]:.6f}")
        res = run_optimization(x0, sigma, vMax, X, ub_f)
        results.append((label, x0, res))

    # Pretty comparison table
    print("\n=== Local Optima Comparison (lower objective is better) ===")
    header = f"{'Start':<14}{'f':>10}{'c':>12}{'d':>10}{'Objective':>14}{'Profit π':>12}{'nit':>8}{'ok?':>6}"
    print(header)
    print("-" * len(header))
    best_idx, best_val = None, np.inf
    for i, (label, x0, res) in enumerate(results):
        if res is None:
            line = f"{label:<14}{'—':>10}{'—':>12}{'—':>10}{'—':>14}{'—':>12}{'—':>8}{'no':>6}"
            print(line)
            continue
        f, c, d = res.x
        obj = res.fun
        prof = -obj
        ok = "yes" if res.success else "no"
        line = f"{label:<14}{f:>10.6f}{c:>12.6f}{d:>10.6f}{obj:>14.6f}{prof:>12.6f}{res.nit:>8d}{ok:>6}"
        print(line)
        if obj < best_val:
            best_val = obj
            best_idx = i

    if best_idx is None:
        print("\n❌ No successful local solutions.")
        return None

    print("\n✅ Chosen best solution is from:", results[best_idx][0])
    return results[best_idx][2]  # return the best scipy OptimizeResult

if __name__ == "__main__":
    print("=" * 60)
    print("THREE-PART TARIFF OPTIMIZATION WITH MULTI-START")
    print("Decision Variables: F, c, d")
    print("=" * 60)

    # Show parameters (unchanged)
    print(f"\nModel Parameters:")
    print(f"  X (market size) = {X}")
    print(f"  σ (quality parameter) = {sigma}")
    print(f"  v_max (max valuation) = {vMax}")
    print(f"  F upper bound = {ub_f}")
    print(f"  d is now a decision variable ∈ [0, 1]")

    # Your existing feasibility/DE-fallback initializer:
    x0_feas = params_init(sigma, vMax, X, ub_f)

    # Compare starts & pick winner
    best_res = compare_starts_and_optimize(x0_feas, sigma, vMax, X, ub_f)

    # Final pretty print + checks
    display_results(best_res, sigma, vMax, X)
