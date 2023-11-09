# ==============================
# Standard Library Imports
# ==============================
from enum import Enum
import random

# ==============================
# Third-party Library Imports
# ==============================
import numpy as np 
import scipy as sp
import pennylane as qml
from autograd import grad, jacobian

# Pennylane numpy
from pennylane import numpy as pnp 

# ==============================
# Import config
# ==============================
import config


def Cost_function(circuit_select):
    """ Calculate Classical-Fisher-Information for qnode(=Post_selection_Dephase).
    
    Args:
        paras (Numpy array): [theta_init, tau_1, tau_2, tau_d1, tau_d2, tau_d3]

    Returns:
        _type_: CFI with minus(-) sign.
    """
    
    phi = pnp.array([config.PHI_GLOBAL], requires_grad = 'True')
          
    CFI = qml.qinfo.classical_fisher(circuit_select)(phi[0])
    
    return -CFI


def run_optimization(sweep_data, initial_parameters, gamma_ps, iterations, circuit_select):
    """ 
    Main function to perform optimization over a range of phi values using the BFGS algorithm.
    
    Args:
        sweep_data (tuple): (start, end, step) values for the phi sweep.
        initial_parameters (numpy_array): Initial parameters for optimization.
        gamma_ps (int): Gamma value for post-selection.
        iterations (int): Number of iterations for the optimization.

    Returns:
        numpy.ndarray: A 3-D array containing phi, CFI, and optimized parameters after each iteration.
    """
    
    # Create Data array
    PHI = np.arange(sweep_data[0], sweep_data[1], sweep_data[2])
    Data = np.zeros((iterations + 1, len(PHI), len(initial_parameters) + 2)) 
    Data[:, :, config.DataIndex.PHI.value] = PHI.squeeze() # Append PHI in to 0th col
    
    # Declare Paras temp 
    Paras_Temporary = 0
    
    # Store initial CFI data and parameters
    for idx, phi in enumerate(PHI):
        Data[config.DataIndex.BEFORE.value][idx][config.DataIndex.CFI.value] = -Cost_function(circuit_select)
        Data[config.DataIndex.BEFORE.value][idx][config.DataIndex.PARAS.value:] = initial_parameters
        
    # Optimize begin
    for iteration in range(1, iterations + 1):
        for phi_idx, phi_current in enumerate(PHI):
            # Determine initial parameters based on the iteration
            if iteration == 1:
                Paras_Temporary = initial_parameters
                
            else:
                Paras_Temporary = Data[iteration][phi_idx][config.DataIndex.PARAS.value:]
            
            # Update the global Phi value
            config.PHI_GLOBAL = phi_current

            # Determine constraints
            constraints = get_constraints(phi_current, gamma_ps, config.Tau_global)

            # Optimize the data
            
            config.Paras_global = Paras_Temporary

            Result_BFGS = BFGS(Paras_Temporary, constraints, circuit_select)
            Data[iteration][phi_idx][config.DataIndex.CFI.value] = -Result_BFGS.fun
            Data[iteration][phi_idx][config.DataIndex.PARAS.value:] = Result_BFGS.x
            
    return Data


def BFGS(initial_parameters, constraints, circuit_select):
    """
    Perform the BFGS optimization algorithm.

    Args:
        initial_parameters (numpy_array): The starting point for the optimization.
        constraints (list of tuple): Bounds on the variables for the optimization.
 
    Returns:
        OptimizeResult: The result of the optimization process.
    """
    gradient = grad(Cost_function)
    hessian = jacobian(gradient)
    
    optimization_result = sp.optimize.minimize(
                fun = Cost_function, 
                x0 = initial_parameters, 
                args = (circuit_select,),
                method = 'L-BFGS-B', 
                bounds = constraints,
                jac = gradient,
                hess = hessian,
                tol = 1e-12,
                options={'ftol': 1e-12, 'gtol': 1e-12}
            )
    return optimization_result


def get_constraints(phi_current, gamma_ps, Tau_global):
    return [(-float('inf'), float('inf'))] * 3