# ==============================
# Standard Library Imports
# ==============================
from enum import Enum


# ==============================
# Third-party Library Imports
# ==============================
import scipy as sp
import numpy as np
import torch
import pennylane as qml
from pennylane import numpy as pnp

import matplotlib.pyplot as plt
from IPython.display import display, Latex
from matplotlib.ticker import MultipleLocator


class INDEX(Enum):
    PHI = 0
    CONCURRENCE = 1
    CFI = 2
    
    PARAS_START = 2
    THETA_1 = 2
    THETA_2 = 3
    PHI_1 = 4
    
def get_bell_density_matrix():
    state_0 = np.array([
        [1],
        [0]
    ])

    state_1 = np.array([
        [0],
        [1]
    ])

    bell_state_vector = (np.kron(state_0, state_0) +  np.kron(state_1, state_1)) / np.sqrt(2) 
    bell_density_matrix = np.kron(bell_state_vector , bell_state_vector.conj().T)

    return bell_density_matrix

def torch_sqrtm(input_matrix):
    """
    Compute the matrix square root of a positive semi-definite matrix in PyTorch.

    Args:
        input_matrix (torch.Tensor): A positive semi-definite matrix.

    Returns:
        torch.Tensor: The matrix square root of the input matrix.
    """
    # Ensure the input is a square matrix
    assert input_matrix.shape[0] == input_matrix.shape[1], "Input matrix must be square."

    # Perform eigen-decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(input_matrix)

    # Compute the square root of eigenvalues
    sqrt_eigenvalues = torch.sqrt(eigenvalues)

    # Reconstruct the square root matrix
    sqrtm = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.transpose(-2, -1)

    return sqrtm

def calculate_concurrence(circuit_select, parameters_in, test_bell=False):
    # Pauli_y matrix
    pauli_y = pnp.array([
        [0, -1.j], 
        [1.j, 0]
    ])
    
    if test_bell == True:
        rho = get_bell_density_matrix()
    else:
        rho = pnp.array(circuit_select(parameters_in))
    rho_unitary = pnp.kron(pauli_y, pauli_y) @ rho.conj() @ pnp.kron(pauli_y, pauli_y)
    
    R = sp.linalg.sqrtm(
        sp.linalg.sqrtm(rho) @ rho_unitary @ sp.linalg.sqrtm(rho)
    )
    
    eig_values = sp.linalg.eigvals(R)
    sort_eig_values = pnp.sort(eig_values.real)[::-1]   # reverse order
    
    sum_of_eig_values = sort_eig_values[0] - pnp.sum(sort_eig_values[1:])
    result_concurrence = pnp.amax(
        [0, sum_of_eig_values]
    )
    
    return result_concurrence

def lbfgsb(cost_function_in, paras_in, gradient_in):
    """
    Optimization using the L-BFGS-B method.

    Args:
        cost_function_in (function): The cost function for optimization.
        paras_in (array): Initial parameters for the optimization.
        constraints_in (list of tuple): Constraints for the optimization parameters.
        gradient_in (function): Gradient of the cost function.

    Returns:
        OptimizeResult: The result of the optimization process.
    """
    
    constraints = [(-float('inf'), float('inf'))] * len(paras_in)
    # constraints[INDEX.PHI_1.value] = [(paras_in[INDEX.PHI_1.value], paras_in[INDEX.PHI_1.value])]   # Fix PHI_SWEEP as it's value

    opt_result = sp.optimize.minimize(
        fun = cost_function_in, 
        x0 = paras_in, 
        method = 'L-BFGS-B', 
        bounds = constraints,
        jac = gradient_in,
        tol = 1e-12,
        options={
            'ftol': 1e-12, 
            'gtol': 1e-12
        }
    )
    return opt_result
    
# def get_data_array(circuit, sweep_range, parameters_in):
#     phi = np.arange(sweep_range[0], sweep_range[1], sweep_range[2])
#     data = np.zeros((len(phi),3))
#     data[:,0] = phi
    
#     if len(parameters_in) == 1:
#         for phi_idx, phi_current in enumerate(phi):
#             data[phi_idx][INDEX.CONCURRENCE.value] = calculate_concurrence(circuit, phi_current)
#             # data[phi_idx][INDEX.CFI.value] = qml.qinfo.classical_fisher(circuit)(pnp.array([phi_current], requires_grad = True))
#     else:
#         for phi_idx, phi_current in enumerate(phi):
#             parameters_in[INDEX.PHI.value] = phi_current
#             data[phi_idx][INDEX.CONCURRENCE.value] = calculate_concurrence(circuit, parameters_in)
            
#     return data

def plot_data(data_array, obeject):
    if obeject == 'CFI':
        plt.plot(data_array[:,INDEX.PHI.value], data_array[:,INDEX.CFI.value], label = '[2-qubit]')
        plt.title(f'CFI')
        plt.xlabel('Time')
        plt.ylabel('CFI')
        plt.grid()
        plt.legend()
        plt.show()
        
    elif obeject == 'CONCURRENCE':
        plt.plot(data_array[:,INDEX.PHI.value], data_array[:, INDEX.CONCURRENCE.value], label = '[2-qubit]')
        plt.title(f'CONCURRENCE')
        plt.xlabel('Time')
        plt.ylabel('CONCURRENCE')
        plt.grid()
        plt.legend()
        plt.show()