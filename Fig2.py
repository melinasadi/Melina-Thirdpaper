import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint, approx_fprime, \
    differential_evolution
from scipy.sparse.linalg import LinearOperator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import parameters
from Main_parameters_value import X, d, sigma, vMax, lb, ub, ub_f

# Import your mathematical functions
from Main_Symbolic import (
    calculate_objective_min,
    numerica_jac_objective_value,
    numerica_hess_objective_value
)

from Main_Fractions import (
    calculate_all_variables,
    verify_beta_sum,
)

from Main_feasibilityregion import find_max_values


# ============================================================================
# INITIALIZATION
# ============================================================================

def params_init(sigma, d, vMax, X, ub_f):
    """Initialize parameters with fallback options"""
    try:
        # Try to find initial values from feasibility analysis
        F_init, c_init, _, _ = find_max_values(X, d, sigma, vMax, ub_f, plot=True, tolerance=0.01)

        if F_init is None or c_init is None:
            raise ValueError("Infeasible result from find_max_values")

        print("✅ Initial values found from feasibility analysis.")
        params_initial = np.array([F_init, c_init])

    except Exception as e:
        # Fallback to global optimization
        print(f"⚠️ Feasibility analysis failed: {e}")
        print("🔁 Using differential evolution for initial guess...")

        # Set up bounds for differential evolution
        bounds_de = [(0.01, ub_f), (0.01, vMax)]

        result = differential_evolution(
            func=lambda x: calculate_objective_min(x[0], x[1], sigma, d, vMax, X),
            bounds=bounds_de,
            strategy='best1bin',
            maxiter=1000,
            polish=True,
            disp=False
        )

        F_init, c_init = result.x
        print("✅ Initial values found using differential evolution.")
        params_initial = np.array([F_init, c_init])

    # Report initial values
    initial_profit = calculate_objective_min(F_init, c_init, sigma, d, vMax, X)
    print(f"Initial point: F = {F_init:.6f}, c = {c_init:.6f}")
    print(f"Initial profit: {-initial_profit:.6f}")

    return params_initial


# ============================================================================
# OBJECTIVE AND DERIVATIVES
# ============================================================================

def objective_function(x, sigma, d, vMax, X):
    """Objective function for minimization"""
    return calculate_objective_min(x[0], x[1], sigma, d, vMax, X)


def jac_obj(x, *args):
    """Jacobian of objective function"""
    sigma, d, vMax, X = args
    F, c = x[0], x[1]

    try:
        # Get jacobian from symbolic calculation
        jac_val = numerica_jac_objective_value(F, c, sigma, d, vMax, X)
        epsilon = np.sqrt(np.finfo(float).eps)

        jac_numeric = approx_fprime(x, objective_function, epsilon, *args)
        jac_diff = jac_val - jac_numeric
        print("jac_diff is,", jac_diff)
        return np.array(jac_val).flatten()
    except Exception as e:
        print(f"Warning: Using numerical gradient due to: {e}")
        epsilon = np.sqrt(np.finfo(float).eps)
        return approx_fprime(x, objective_function, epsilon, *args)


def hess_obj(x, *args):
    """Hessian of objective function as LinearOperator"""
    sigma, d, vMax, X = args
    F, c = x[0], x[1]

    def matvec(v):
        try:
            H = numerica_hess_objective_value(F, c, sigma, d, vMax, X)
            return np.dot(H, v)
        except:
            return v  # Fallback to identity

    return LinearOperator((len(x), len(x)), matvec=matvec)


# ============================================================================
# CONSTRAINT FUNCTIONS
# ============================================================================

def constraint_func3(x, sigma, d, vMax, X):
    """Beta_P constraint: 0 <= beta_P <= 1"""
    beta_p, _, _, _, _ = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return beta_p


def constraint_func4(x, sigma, d, vMax, X):
    """Beta_R constraint: 0 <= beta_R <= 1"""
    _, beta_r, _, _, _ = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return beta_r


def constraint_func5(x, sigma, d, vMax, X):
    """Beta_N constraint: 0 <= beta_N <= 1"""
    _, _, beta_n, _, _ = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return beta_n


def constraint_func6(x, sigma, d, vMax, X):
    """Sum of betas should equal 1"""
    beta_p, beta_r, beta_n, _, _ = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return (beta_p + beta_r + beta_n) - 1


def constraint_func7(x, sigma, d, vMax, X):
    """Delta_P constraint: delta_P >= 0"""
    _, _, _, delta_p, _ = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return delta_p


def constraint_func8(x, sigma, d, vMax, X):
    """Delta_R constraint: delta_R >= 0"""
    _, _, _, _, delta_r = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)
    return delta_r


# ============================================================================
# SETUP CONSTRAINTS
# ============================================================================

def setup_constraints(sigma, d, vMax, X):
    """Set up all constraints for the optimization problem"""
    constraints = []

    # Linear constraints
    # Constraint 1: 0 <= c <= vMax
    A1 = np.array([[0, 1]])  # [0*F + 1*c]
    constraint1 = LinearConstraint(A1, [0], [vMax])
    constraints.append(constraint1)

    # Constraint 2: F <= X * (d + sigma - 1) * c
    A2 = np.array([[1, -X * (d + sigma - 1)]])  # [1*F - coeff*c]
    constraint2 = LinearConstraint(A2, [-np.inf], [0])
    constraints.append(constraint2)

    # Constraint 3: F >= 0
    A3 = np.array([[1, 0]])  # [1*F + 0*c]
    constraint3 = LinearConstraint(A3, [0], [np.inf])
    constraints.append(constraint3)

    # Nonlinear constraints
    # Beta constraints (0 <= beta <= 1)
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func3(x, sigma, d, vMax, X), lb=0, ub=1))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func4(x, sigma, d, vMax, X), lb=0, ub=1))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func5(x, sigma, d, vMax, X), lb=0, ub=1))

    # Sum of betas = 1
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func6(x, sigma, d, vMax, X), lb=0, ub=0))

    # Delta constraints (>= 0)
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func7(x, sigma, d, vMax, X), lb=0, ub=np.inf))
    constraints.append(NonlinearConstraint(
        lambda x: constraint_func8(x, sigma, d, vMax, X), lb=0, ub=np.inf))

    return constraints


# ============================================================================
# OPTIMIZATION FUNCTION
# ============================================================================

def run_optimization(params_init, sigma, d, vMax, X, ub_f):
    """Run the optimization"""
    # Set up bounds and constraints
    bounds = Bounds([0, 0], [ub_f, vMax])
    constraints = setup_constraints(sigma, d, vMax, X)
    args = (sigma, d, vMax, X)

    print("\n=== Starting Optimization ===")

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
                'initial_tr_radius': 2,
                'gtol': 1e-15,
                'xtol': 1e-15,
                'barrier_tol': 1e-15,
                'maxiter': 2000,
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

def check_constraints(x, sigma, d, vMax, X):
    """Check if all constraints are satisfied"""
    print("\n=== Constraint Verification ===")

    # Linear constraints
    print("\nLinear constraints:")
    print(f"  c = {x[1]:.6f} (should be in [0, {vMax}]): {'✓' if 0 <= x[1] <= vMax else '✗'}")
    print(f"  F = {x[0]:.6f} (should be >= 0): {'✓' if x[0] >= 0 else '✗'}")

    F_max = X * (d + sigma - 1) * x[1]
    print(f"  F <= {F_max:.6f}: {'✓' if x[0] <= F_max + 1e-6 else '✗'}")

    # Nonlinear constraints
    print("\nNonlinear constraints:")
    beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(x[0], x[1], sigma, d, vMax, X)

    print(f"  β^P = {beta_p:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_p <= 1 else '✗'}")
    print(f"  β^R = {beta_r:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_r <= 1 else '✗'}")
    print(f"  β^N = {beta_n:.6f} (should be in [0, 1]): {'✓' if 0 <= beta_n <= 1 else '✗'}")

    beta_sum = beta_p + beta_r + beta_n
    print(f"  Sum of betas = {beta_sum:.8f} (should be 1.0): {'✓' if abs(beta_sum - 1) < 1e-6 else '✗'}")

    print(f"  δ^P = {delta_p:.6f} (should be >= 0): {'✓' if delta_p >= -1e-6 else '✗'}")
    print(f"  δ^R = {delta_r:.6f} (should be >= 0): {'✓' if delta_r >= -1e-6 else '✗'}")


def display_results(result, sigma, d, vMax, X):
    """Display optimization results"""
    if result is None or not result.success:
        print("\nOptimization failed!")
        return None, None

    F_opt, c_opt = result.x
    optimal_profit = -result.fun  # Convert back to positive (actual profit)

    print("\n=== Optimization Results ===")
    print(f"Optimal F: {F_opt:.6f}")
    print(f"Optimal c: {c_opt:.6f}")
    print(f"Optimal profit: {optimal_profit:.6f}")

    # Calculate all variables at optimal point
    beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(F_opt, c_opt, sigma, d, vMax, X)

    print(f"\nOptimal Variables:")
    print(f"  β^P = {beta_p:.6f}")
    print(f"  β^R = {beta_r:.6f}")
    print(f"  β^N = {beta_n:.6f}")
    print(f"  δ^P = {delta_p:.6f}")
    print(f"  δ^R = {delta_r:.6f}")

    # Verify profit calculation
    verify_profit = -calculate_objective_min(F_opt, c_opt, sigma, d, vMax, X)
    print(f"\nProfit verification: {verify_profit:.6f} (should match {optimal_profit:.6f})")

    # Check constraints
    check_constraints(result.x, sigma, d, vMax, X)

    return result.x, optimal_profit


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def boundary_1(v, F, c, d, sigma):
    return F / ((sigma * v) - ((1 - d) * c))


def boundary_2(v, F, c, d, sigma):
    return F / (((sigma - 1) * v) + (d * c))


def boundary_3(v, F, c, d, sigma, X_grid):
    return (X_grid * (1 - d) * c + F) / (sigma * X_grid)


def plot_customer_regions(F, c, sigma, d, vMax, X):
    """Plot customer preference regions"""
    # Define the range for plotting lines
    v1 = np.linspace(0.0001, c, 400)
    x1 = boundary_1(v1, F, c, d, sigma)
    valid_mask_1_line = (sigma * v1 - (1 - d) * c) > 0
    v1 = v1[valid_mask_1_line]
    x1 = x1[valid_mask_1_line]

    v2 = np.linspace(c, vMax, 400)
    x2 = boundary_2(v2, F, c, d, sigma)

    # Create a grid for regions
    v_grid, X_grid = np.meshgrid(np.linspace(0, vMax, 400), np.linspace(0.001, X, 400))

    # Compute boundaries with valid masks
    boundary_1_valid = boundary_1(v_grid, F, c, d, sigma)
    valid_mask_1 = (sigma * v_grid - (1 - d) * c) > 0
    invalid_mask_1 = ~valid_mask_1
    boundary_1_valid[invalid_mask_1] = np.nan

    boundary_3_valid = boundary_3(v_grid, F, c, d, sigma, X_grid)

    valid_mask_3 = (sigma * X_grid) != 0  # Avoid division by zero
    boundary_3_valid[~valid_mask_3] = np.nan  # Set invalid values to NaN

    # Define regions with valid masks
    region_nonusers = (
            ((X_grid < boundary_1_valid) & valid_mask_1 & (v_grid <= c)) |
            (invalid_mask_1 & (v_grid <= c)) |
            ((X_grid > boundary_3_valid) & valid_mask_3 & (v_grid <= c))
    )

    region_regular = (X_grid <= boundary_2(v_grid, F, c, d, sigma)) & (v_grid >= c)

    region_premium = (
            ((X_grid > boundary_2(v_grid, F, c, d, sigma)) & (v_grid >= c)) |
            ((X_grid > boundary_1_valid) & valid_mask_1 & (v_grid <= c))
    )

    # Create a category array
    category = np.zeros_like(X_grid, dtype=int)
    category[region_nonusers] = 0  # Nonusers
    category[region_regular] = 1  # Regular users
    category[region_premium] = 2  # Premium users

    # Define colors
    cmap = ListedColormap(['green', 'orange', 'blue'])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    # Plot the figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot regions
    contour = ax.contourf(v_grid, X_grid, category, levels=[-0.5, 0.5, 1.5, 2.5],
                          cmap=cmap, norm=norm, alpha=0.7)

    # Plot boundary lines
    ax.plot(v1, x1, color='black', label=r'$x = \frac{F}{\sigma v - (1-d)c}$')
    ax.plot(v2, x2, color='black', label=r'$x = \frac{F}{(\sigma - 1)v + dc}$')

    # Mark critical points
    if len(v1) > 0:  # Check if v1 is not empty
        ax.scatter([c], [F / (sigma * c - (1 - d) * c)], color='black', zorder=5)

    # Add labels and title
    # ax.set_title("Customers' Subscription Preferences Based on Usage Frequency and Valuation")
    ax.set_xlabel("v",fontsize=16)
    ax.set_ylabel("x",fontsize=16)
    ax.set_xlim(0, vMax)
    ax.set_ylim(0, X)

    # Add legend
    legend_handles = [
        Patch(facecolor='green', edgecolor='black', label='Nonusers'),
        Patch(facecolor='orange', edgecolor='black', label='Regular Users'),
        Patch(facecolor='blue', edgecolor='black', label='Premium Users'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    # Show grid and plot
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TWO-PART TARIFF OPTIMIZATION")
    print("=" * 60)

    # Display parameters
    print(f"\nModel Parameters:")
    print(f"  X (market size) = {X}")
    print(f"  d (discount rate) = {d}")
    print(f"  σ (quality parameter) = {sigma}")
    print(f"  v_max (max valuation) = {vMax}")
    print(f"  F upper bound = {ub_f}")

    # Initialize parameters
    params_initial = params_init(sigma, d, vMax, X, ub_f)

    # Verify initial point
    print("\nVerifying initial point...")
    verify_beta_sum(params_initial[0], params_initial[1], sigma, d, vMax, X)

    # Run optimization
    result = run_optimization(params_initial, sigma, d, vMax, X, ub_f)

    # Display results and get optimal parameters
    optimal_params = None
    optimal_value = None

    if result is not None:
        optimal_params, optimal_value = display_results(result, sigma, d, vMax, X)

        # Try alternative method if failed
        if not result.success:
            print("\n" + "=" * 60)
            print("Trying alternative optimization method (SLSQP)...")
            print("=" * 60)

            # Set up for SLSQP
            bounds = Bounds([0, 0], [ub_f, vMax])
            constraints = setup_constraints(sigma, d, vMax, X)
            args = (sigma, d, vMax, X)

            result_alt = minimize(
                fun=objective_function,
                x0=params_initial,
                args=args,
                method='SLSQP',
                jac=jac_obj,
                bounds=bounds,
                constraints=constraints,
                options={
                    'ftol': 1e-9,
                    'maxiter': 1000,
                    'disp': True
                }
            )

            if result_alt.success:
                print("\nSLSQP succeeded!")
                optimal_params, optimal_value = display_results(result_alt, sigma, d, vMax, X)

    # Plot results if optimization succeeded
    if optimal_params is not None:
        print("\n" + "=" * 60)
        print("PLOTTING CUSTOMER PREFERENCE REGIONS")
        print("=" * 60)

        F = optimal_params[0]
        c = optimal_params[1]

        # Calculate and display the optimal variables
        beta_p, beta_r, beta_n, delta_p, delta_r = calculate_all_variables(F, c, sigma, d, vMax, X)
        print(f"\nOptimal parameters: F = {F:.6f}, c = {c:.6f}")
        print(f"Optimal objective function value: {optimal_value:.6f}")
        print(f"β^P = {beta_p:.6f}, β^R = {beta_r:.6f}, β^N = {beta_n:.6f}")
        print(f"δ^P = {delta_p:.6f}, δ^R = {delta_r:.6f}")

        # Plot the customer preference regions
        plot_customer_regions(F, c, sigma, d, vMax, X)
    else:
        print("\nOptimization failed - cannot plot results.")