import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from S_parameters_value import X, sigma, vMax, ub_f  # Note: d is no longer imported as parameter

# Import from updated symbolic file with 3 variables
from S_Symbolic import (
    calculate_beta_p, calculate_beta_r, calculate_beta_n,
    calculate_delta_p, calculate_delta_r, calculate_omega, calculate_gamma,
    calculate_objective_min
)


def calculate_objective_clean(x, sigma, vMax, X):
    """
    Calculate objective with 3 decision variables
    x = [F, c, d]
    """
    F, c, d = x[0], x[1], x[2]
    # Note: calculate_objective_min returns negative profit for minimization
    objective_value = -calculate_objective_min(F, c, d, sigma, vMax, X)
    return objective_value


def calculate_variables_clean(x, sigma, vMax, X):
    """
    Calculate all variables with 3 decision variables
    x = [F, c, d]
    """
    F, c, d = x[0], x[1], x[2]

    beta_p = calculate_beta_p(F, c, d, sigma, vMax, X)
    beta_r = calculate_beta_r(F, c, d, sigma, vMax, X)
    beta_n = calculate_beta_n(F, c, d, sigma, vMax, X)
    delta_p = calculate_delta_p(F, c, d, sigma, vMax, X)
    delta_r = calculate_delta_r(F, c, d, sigma, vMax, X)

    # Calculate common_log_term2 for compatibility
    gamma = calculate_gamma(F, c, d, sigma, vMax, X)
    omega = calculate_omega(F, c, d, sigma, vMax, X)
    common_log_term2 = np.log(gamma / (c * omega))

    return beta_p, beta_r, beta_n, delta_p, delta_r, common_log_term2


def summation_check(x, X, sigma, vMax):
    """Check that beta fractions sum to 1"""
    beta_t_p, beta_t_r, beta_t_n, _, _, _ = calculate_variables_clean(x, sigma, vMax, X)
    summation = (beta_t_p + beta_t_r + beta_t_n) - 1
    return summation


def calculate_F_bounds(X, c, sigma, d, v_max):
    """Calculate feasible bounds for F parameter given c and d"""
    denominator = sigma * c + c * d - v_max
    if abs(denominator) < 1e-10:  # Avoid division by zero
        return 0.01, X * c  # Default bounds

    base_value = (X * c * (sigma + d - 1) * (sigma * v_max + c * d - v_max)) / denominator
    sqrt_term = sigma * c * (sigma + d - 1) * (sigma * v_max + c * d - v_max)

    if sqrt_term < 0:  # Avoid complex numbers
        return 0.01, X * c  # Default bounds

    sqrt_value = (X * np.sqrt(sqrt_term) * (c - v_max)) / denominator
    F_lower = min(base_value - sqrt_value, base_value + sqrt_value)
    F_upper = max(base_value - sqrt_value, base_value + sqrt_value)

    # Ensure bounds are positive
    F_lower = max(0.01, F_lower)
    F_upper = max(F_lower + 0.01, F_upper)

    return F_lower, F_upper


def find_max_values_3d(X, sigma, vMax, ub_f, plot=True, tolerance=0.01,
                       n_F=50, n_c=40, n_d=30):
    """
    Find maximum values using brute force search in 3D space

    Parameters:
    X, sigma, vMax: Model parameters
    ub_f: Upper bound for F
    plot: Whether to show plots
    tolerance: Tolerance for finding flat regions
    n_F, n_c, n_d: Number of grid points for each variable
    """

    # Define search bounds
    F_vals = np.linspace(0.01, ub_f, n_F)
    c_vals = np.linspace(0.5, vMax, n_c)
    d_vals = np.linspace(0.01, 0.99, n_d)  # d should be between 0 and 1

    # Initialize storage for results
    max_objective = -np.inf
    optimal_point = None
    feasible_points = []
    all_objectives = []

    print(f"Searching {n_F * n_c * n_d} points in 3D space...")

    # Progress tracking
    total_points = n_F * n_c * n_d
    points_checked = 0

    # Brute force search
    for i, F in enumerate(F_vals):
        for j, c in enumerate(c_vals):
            for k, d in enumerate(d_vals):
                points_checked += 1

                # Progress indicator
                if points_checked % 10000 == 0:
                    print(f"Progress: {points_checked}/{total_points} ({100 * points_checked / total_points:.1f}%)")

                try:
                    # Calculate F bounds for feasibility check
                    F_lower, F_upper = calculate_F_bounds(X, c, sigma, d, vMax)

                    # Check if F is within feasible bounds
                    if F > F_upper or F < 0:
                        continue

                    # Calculate objective
                    obj_val = calculate_objective_clean([F, c, d], sigma, vMax, X)

                    # Calculate variables for feasibility check
                    beta_p, beta_r, beta_n, delta_p, delta_r, _ = calculate_variables_clean(
                        [F, c, d], sigma, vMax, X
                    )

                    # Check feasibility constraints
                    if (0 <= c <= vMax) and (obj_val > 0) and (0 <= F <= F_upper):
                        if (0 <= beta_p <= 1) and (0 <= beta_r <= 1) and (0 <= beta_n <= 1):
                            if (delta_p >= 0) and (delta_r >= 0):
                                # Check sum constraint
                                sum_betas = beta_p + beta_r + beta_n
                                if abs(sum_betas - 1.0) < 0.01:  # Allow small tolerance
                                    feasible_points.append((F, c, d, obj_val))
                                    all_objectives.append(obj_val)

                                    if obj_val > max_objective:
                                        max_objective = obj_val
                                        optimal_point = (F, c, d)

                except Exception as e:
                    continue

    print(f"\nFound {len(feasible_points)} feasible points")

    if not feasible_points:
        print("Warning: No feasible points found!")
        return None, None, None, None, None

    # Find points in flat region (near-optimal)
    flat_region_points = []
    for point in feasible_points:
        if point[3] >= max_objective - tolerance:
            flat_region_points.append(point[:3])  # Store (F, c, d)

    print(f"Found {len(flat_region_points)} near-optimal points")

    # Plotting
    if plot and feasible_points:
        create_plots_3d(feasible_points, flat_region_points, max_objective)

    # Return results
    if optimal_point:
        opt_F, opt_c, opt_d = optimal_point
        print(f"\n=== Optimal Solution ===")
        print(f"F* = {opt_F:.6f}")
        print(f"c* = {opt_c:.6f}")
        print(f"d* = {opt_d:.6f}")
        print(f"Max objective value = {max_objective:.6f}")

        return opt_F, opt_c, opt_d, max_objective, flat_region_points

    return None, None, None, None, None


def create_plots_3d(feasible_points, flat_region_points, max_objective):
    """Create visualization plots for 3D optimization results"""

    fig = plt.figure(figsize=(20, 12))

    # Extract data for plotting
    F_feas = [p[0] for p in feasible_points]
    c_feas = [p[1] for p in feasible_points]
    d_feas = [p[2] for p in feasible_points]
    obj_feas = [p[3] for p in feasible_points]

    # 1. 3D scatter plot colored by objective value
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter1 = ax1.scatter(F_feas, c_feas, d_feas, c=obj_feas, cmap='viridis',
                           s=20, alpha=0.6)
    if flat_region_points:
        F_opt = [p[0] for p in flat_region_points]
        c_opt = [p[1] for p in flat_region_points]
        d_opt = [p[2] for p in flat_region_points]
        ax1.scatter(F_opt, c_opt, d_opt, color='red', s=100, marker='*',
                    label='Near-optimal points')
    ax1.set_xlabel('F')
    ax1.set_ylabel('c')
    ax1.set_zlabel('d')
    ax1.set_title('3D Feasible Region')
    plt.colorbar(scatter1, ax=ax1, label='Objective Value')
    if flat_region_points:
        ax1.legend()

    # 2. F vs c projection (averaging over d)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter2 = ax2.scatter(F_feas, c_feas, c=obj_feas, cmap='viridis',
                           s=20, alpha=0.6)
    if flat_region_points:
        ax2.scatter(F_opt, c_opt, color='red', s=50, marker='*',
                    label='Near-optimal')
    ax2.set_xlabel('F')
    ax2.set_ylabel('c')
    ax2.set_title('F vs c Projection')
    plt.colorbar(scatter2, ax=ax2, label='Objective Value')
    if flat_region_points:
        ax2.legend()

    # 3. F vs d projection
    ax3 = fig.add_subplot(2, 3, 3)
    scatter3 = ax3.scatter(F_feas, d_feas, c=obj_feas, cmap='viridis',
                           s=20, alpha=0.6)
    if flat_region_points:
        ax3.scatter(F_opt, d_opt, color='red', s=50, marker='*',
                    label='Near-optimal')
    ax3.set_xlabel('F')
    ax3.set_ylabel('d')
    ax3.set_title('F vs d Projection')
    plt.colorbar(scatter3, ax=ax3, label='Objective Value')
    if flat_region_points:
        ax3.legend()

    # 4. c vs d projection
    ax4 = fig.add_subplot(2, 3, 4)
    scatter4 = ax4.scatter(c_feas, d_feas, c=obj_feas, cmap='viridis',
                           s=20, alpha=0.6)
    if flat_region_points:
        ax4.scatter(c_opt, d_opt, color='red', s=50, marker='*',
                    label='Near-optimal')
    ax4.set_xlabel('c')
    ax4.set_ylabel('d')
    ax4.set_title('c vs d Projection')
    plt.colorbar(scatter4, ax=ax4, label='Objective Value')
    if flat_region_points:
        ax4.legend()

    # 5. Objective value distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(obj_feas, bins=50, edgecolor='black', alpha=0.7)
    ax5.axvline(max_objective, color='red', linestyle='--', linewidth=2,
                label=f'Max: {max_objective:.4f}')
    ax5.set_xlabel('Objective Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Objective Value Distribution')
    ax5.legend()

    # 6. Convergence to optimal (sorted objectives)
    ax6 = fig.add_subplot(2, 3, 6)
    sorted_obj = sorted(obj_feas)
    ax6.plot(range(len(sorted_obj)), sorted_obj, 'b-', alpha=0.7)
    ax6.axhline(max_objective, color='red', linestyle='--', linewidth=2,
                label=f'Max: {max_objective:.4f}')
    ax6.set_xlabel('Point Index (sorted)')
    ax6.set_ylabel('Objective Value')
    ax6.set_title('Sorted Objective Values')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_optimal_solution(F_opt, c_opt, d_opt, sigma, vMax, X):
    """Analyze the optimal solution in detail"""

    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS OF OPTIMAL SOLUTION")
    print("=" * 60)

    # Calculate all variables at optimal point
    x_opt = [F_opt, c_opt, d_opt]
    beta_p, beta_r, beta_n, delta_p, delta_r, _ = calculate_variables_clean(
        x_opt, sigma, vMax, X
    )
    obj_val = calculate_objective_clean(x_opt, sigma, vMax, X)

    print(f"\n=== Decision Variables ===")
    print(f"F* = {F_opt:.6f}")
    print(f"c* = {c_opt:.6f}")
    print(f"d* = {d_opt:.6f}")

    print(f"\n=== Beta Values ===")
    print(f"β^P = {beta_p:.6f}")
    print(f"β^R = {beta_r:.6f}")
    print(f"β^N = {beta_n:.6f}")
    print(f"Sum of betas = {beta_p + beta_r + beta_n:.8f} (should be 1.0)")

    print(f"\n=== Delta Values ===")
    print(f"δ^P = {delta_p:.6f}")
    print(f"δ^R = {delta_r:.6f}")

    print(f"\n=== Objective Function ===")
    print(f"Profit (π) = {obj_val:.6f}")

    # Calculate bounds for F
    F_lower, F_upper = calculate_F_bounds(X, c_opt, sigma, d_opt, vMax)
    print(f"\n=== Feasibility Check ===")
    print(f"F bounds: [{F_lower:.6f}, {F_upper:.6f}]")
    print(f"F* within bounds: {F_lower <= F_opt <= F_upper}")
    print(f"c* within bounds: {0 <= c_opt <= vMax}")
    print(f"d* within bounds: {0 <= d_opt <= 1}")

    return {
        'F': F_opt, 'c': c_opt, 'd': d_opt,
        'beta_p': beta_p, 'beta_r': beta_r, 'beta_n': beta_n,
        'delta_p': delta_p, 'delta_r': delta_r,
        'objective': obj_val
    }


# if __name__ == "__main__":
#     # # Set parameters
#     # X = 10.0
#     # sigma = 1.2
#     # vMax = 1.0
#     # ub_f = 500.0  # Upper bound for F
#     #
#     # print("=== 3D Brute Force Optimization ===")
#     # print(f"Parameters: X={X}, sigma={sigma}, vMax={vMax}")
#     # print(f"Search bounds: F∈[0.01, {ub_f}], c∈[0.5, {vMax}], d∈[0.01, 0.99]")
#
#     # Run optimization with reasonable grid resolution
#     # Increase n_F, n_c, n_d for more accuracy (but slower computation)
#     F_opt, c_opt, d_opt, max_obj, near_optimal = find_max_values_3d(
#         X, sigma, vMax, ub_f,
#         plot=True,
#         tolerance=0.01,
#         n_F=30,  # Number of F grid points
#         n_c=25,  # Number of c grid points
#         n_d=20  # Number of d grid points
#     )
#
#     if F_opt is not None:
#         # Detailed analysis
#         results = analyze_optimal_solution(F_opt, c_opt, d_opt, sigma, vMax, X)
#
#         # Save results
#         print("\n=== Saving Results ===")
#         np.savez('brute_force_results_3d.npz',
#                  F_opt=F_opt, c_opt=c_opt, d_opt=d_opt,
#                  max_objective=max_obj,
#                  results=results,
#                  near_optimal_points=near_optimal)