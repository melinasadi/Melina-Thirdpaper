import numpy as np
from Main_Symbolic import (numerical_jac_beta_t_p, numerical_hess_beta_t_p, numerical_jac_beta_t_r,
                           numerical_hess_beta_t_r, numerical_jac_beta_t_n, numerical_hess_beta_t_n,
                           numeircal_hessian_delta_t_r, numerical_jacobian_delta_t_p ,
                           numerical_hessian_delta_t_p, numerical_jacobian_delta_t_r)

from Main_parameters_value import args
from scipy.sparse.linalg import LinearOperator

def constraint_jac3(x):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    jac_aggregated = np.zeros((T + 1, 2), dtype=np.float64)  # Ensure high precision

    for t in range(1, T + 1):
        jac_val = numerical_jac_beta_t_p(x[0], x[t], sigma, d, vMax, X)
        if np.any(np.isnan(jac_val)) or np.any(np.isinf(jac_val)):
            raise ValueError(f"Invalid Jacobian value at t={t}: {jac_val}")
        jac_aggregated[t, :] = jac_val  # Direct assignment for stability

    jac_final = np.zeros(len(x), dtype=np.float64)
    jac_final[0] = np.sum(jac_aggregated[1:, 0])  # Sum all partial derivatives w.r.t x[0]
    jac_final[1:] = jac_aggregated[1:, 1]  # Copy the partial derivative w.r.t x[t] directly

    # Optional: Apply a small smoothing technique if noise persists
    jac_final = np.round(jac_final, decimals=8)  # Adjust the precision as needed

    return jac_final


# def constraint_hess3(x,v):
#     # print("Input x3:", x)
#     _, X, _, d, sigma, vMax, T = args
#     T = len(x) - 1
#     hess_aggregated = np.zeros((T + 1, 2))
#     for t in range(1, T + 1):
#         hess_val = numerical_hess_beta_t_p(x[0], x[t], sigma, d, vMax, X)
#         # print('hess_val3 is: ',hess_val)
#         hess_flat = np.array(hess_val).flatten()
#         hess_aggregated[t, 0] = hess_flat[0]  # Assuming the first value is derivative w.r.t x[0]
#         hess_aggregated[t, 1] = hess_flat[1]
#     hess_vector = hess_aggregated.flatten()
#     hess_final = np.zeros(len(x))
#     hess_final[0] = np.sum(hess_aggregated[:, 0])  # Sum all partial derivatives w.r.t x[0]
#     for t in range(1, T + 1):
#         hess_final[t] = hess_aggregated[t, 1]  # Copy the partial derivative w.r.t x[t]
#     # print('hess_final3 is: ',hess_final)
#     return hess_final

def constraint_hess3(x, v):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    hess_matrix = np.zeros((T + 1, T + 1))  # Hessian should be a (T+1) x (T+1) matrix

    for t in range(1, T + 1):
        # Compute the Hessian value for each time step
        hess_val = numerical_hess_beta_t_p(x[0], x[t], sigma, d, vMax, X)
        hess_val = np.array(hess_val)

        # Aggregate into the Hessian matrix
        hess_matrix[0, 0] += hess_val[0, 0]  # Sum partial derivatives w.r.t x[0]
        hess_matrix[t, 0] = hess_val[1, 0]  # Cross partial derivatives
        hess_matrix[0, t] = hess_val[1, 0]  # Cross partial derivatives (symmetric)
        hess_matrix[t, t] = hess_val[1, 1]  # Partial derivative w.r.t x[t]

    return hess_matrix

def constraint_jac4(x):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    jac_aggregated = np.zeros((T + 1, 2), dtype=np.float64)  # Ensure high precision

    for t in range(1, T + 1):
        jac_val = numerical_jac_beta_t_r(x[0], x[t], sigma, d, vMax, X)
        if np.any(np.isnan(jac_val)) or np.any(np.isinf(jac_val)):
            raise ValueError(f"Invalid Jacobian value at t={t}: {jac_val}")
        jac_aggregated[t, :] = jac_val  # Direct assignment for stability

    jac_final = np.zeros(len(x), dtype=np.float64)
    jac_final[0] = np.sum(jac_aggregated[1:, 0])  # Sum all partial derivatives w.r.t x[0]
    jac_final[1:] = jac_aggregated[1:, 1]  # Copy the partial derivative w.r.t x[t] directly

    # Optional: Apply a small smoothing technique if noise persists
    jac_final = np.round(jac_final, decimals=8)  # Adjust the precision as needed

    return jac_final

def constraint_hess4(x, v):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    hess_matrix = np.zeros((T + 1, T + 1))  # Hessian should be a (T+1) x (T+1) matrix

    for t in range(1, T + 1):
        # Compute the Hessian value for each time step
        hess_val = numerical_hess_beta_t_r(x[0], x[t], sigma, d, vMax, X)
        hess_val = np.array(hess_val)

        # Aggregate into the Hessian matrix
        hess_matrix[0, 0] += hess_val[0, 0]  # Sum partial derivatives w.r.t x[0]
        hess_matrix[t, 0] = hess_val[1, 0]  # Cross partial derivatives
        hess_matrix[0, t] = hess_val[1, 0]  # Cross partial derivatives (symmetric)
        hess_matrix[t, t] = hess_val[1, 1]  # Partial derivative w.r.t x[t]

    return hess_matrix

def constraint_jac5(x):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    jac_aggregated = np.zeros((T + 1, 2), dtype=np.float64)  # Ensure high precision

    for t in range(1, T + 1):
        jac_val = numerical_jac_beta_t_n(x[0], x[t], sigma, d, vMax, X)
        if np.any(np.isnan(jac_val)) or np.any(np.isinf(jac_val)):
            raise ValueError(f"Invalid Jacobian value at t={t}: {jac_val}")
        jac_aggregated[t, :] = jac_val  # Direct assignment for stability

    jac_final = np.zeros(len(x), dtype=np.float64)
    jac_final[0] = np.sum(jac_aggregated[1:, 0])  # Sum all partial derivatives w.r.t x[0]
    jac_final[1:] = jac_aggregated[1:, 1]  # Copy the partial derivative w.r.t x[t] directly

    # Optional: Apply a small smoothing technique if noise persists
    jac_final = np.round(jac_final, decimals=8)  # Adjust the precision as needed

    return jac_final


def constraint_hess5(x, v):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    hess_matrix = np.zeros((T + 1, T + 1))  # Hessian should be a (T+1) x (T+1) matrix

    for t in range(1, T + 1):
        # Compute the Hessian value for each time step
        hess_val = numerical_hess_beta_t_n(x[0], x[t], sigma, d, vMax, X)
        hess_val = np.array(hess_val)

        # Aggregate into the Hessian matrix
        hess_matrix[0, 0] += hess_val[0, 0]  # Sum partial derivatives w.r.t x[0]
        hess_matrix[t, 0] = hess_val[1, 0]  # Cross partial derivatives
        hess_matrix[0, t] = hess_val[1, 0]  # Cross partial derivatives (symmetric)
        hess_matrix[t, t] = hess_val[1, 1]  # Partial derivative w.r.t x[t]

    return hess_matrix


def constraint_jac7(x):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    jac_aggregated = np.zeros((T + 1, 2), dtype=np.float64)  # Ensure high precision

    for t in range(1, T + 1):
        jac_val = numerical_jacobian_delta_t_r(x[0], x[t], sigma, d, vMax, X)
        if np.any(np.isnan(jac_val)) or np.any(np.isinf(jac_val)):
            raise ValueError(f"Invalid Jacobian value at t={t}: {jac_val}")
        jac_aggregated[t, :] = jac_val  # Direct assignment for stability

    jac_final = np.zeros(len(x), dtype=np.float64)
    jac_final[0] = np.sum(jac_aggregated[1:, 0])  # Sum all partial derivatives w.r.t x[0]
    jac_final[1:] = jac_aggregated[1:, 1]  # Copy the partial derivative w.r.t x[t] directly

    # Optional: Apply a small smoothing technique if noise persists
    jac_final = np.round(jac_final, decimals=8)  # Adjust the precision as needed

    return jac_final


def constraint_hess7(x, v):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    hess_matrix = np.zeros((T + 1, T + 1))  # Hessian should be a (T+1) x (T+1) matrix

    for t in range(1, T + 1):
        # Compute the Hessian value for each time step
        hess_val = numeircal_hessian_delta_t_r(x[0], x[t], sigma, d, vMax, X)
        hess_val = np.array(hess_val)

        # Aggregate into the Hessian matrix
        hess_matrix[0, 0] += hess_val[0, 0]  # Sum partial derivatives w.r.t x[0]
        hess_matrix[t, 0] = hess_val[1, 0]  # Cross partial derivatives
        hess_matrix[0, t] = hess_val[1, 0]  # Cross partial derivatives (symmetric)
        hess_matrix[t, t] = hess_val[1, 1]  # Partial derivative w.r.t x[t]

    return hess_matrix




def constraint_jac8(x):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    jac_aggregated = np.zeros((T + 1, 2), dtype=np.float64)  # Ensure high precision

    for t in range(1, T + 1):
        jac_val = numerical_jacobian_delta_t_p(x[0], x[t], sigma, d, vMax, X)
        if np.any(np.isnan(jac_val)) or np.any(np.isinf(jac_val)):
            raise ValueError(f"Invalid Jacobian value at t={t}: {jac_val}")
        jac_aggregated[t, :] = jac_val  # Direct assignment for stability

    jac_final = np.zeros(len(x), dtype=np.float64)
    jac_final[0] = np.sum(jac_aggregated[1:, 0])  # Sum all partial derivatives w.r.t x[0]
    jac_final[1:] = jac_aggregated[1:, 1]  # Copy the partial derivative w.r.t x[t] directly

    # Optional: Apply a small smoothing technique if noise persists
    jac_final = np.round(jac_final, decimals=8)  # Adjust the precision as needed

    return jac_final



def constraint_hess8(x, v):
    # Unpack the arguments
    _, X, _, d, sigma, vMax, _ = args
    T = len(x) - 1
    hess_matrix = np.zeros((T + 1, T + 1))  # Hessian should be a (T+1) x (T+1) matrix

    for t in range(1, T + 1):
        # Compute the Hessian value for each time step
        hess_val = numerical_hessian_delta_t_p(x[0], x[t], sigma, d, vMax, X)
        hess_val = np.array(hess_val)

        # Aggregate into the Hessian matrix
        hess_matrix[0, 0] += hess_val[0, 0]  # Sum partial derivatives w.r.t x[0]
        hess_matrix[t, 0] = hess_val[1, 0]  # Cross partial derivatives
        hess_matrix[0, t] = hess_val[1, 0]  # Cross partial derivatives (symmetric)
        hess_matrix[t, t] = hess_val[1, 1]  # Partial derivative w.r.t x[t]

    return hess_matrix



