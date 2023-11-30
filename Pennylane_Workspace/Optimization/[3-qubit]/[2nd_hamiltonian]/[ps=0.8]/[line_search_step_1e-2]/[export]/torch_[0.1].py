
# ==============================
# Standard Library Imports
# ==============================
from enum import Enum
import random

# ==============================
# Third-party Library Imports
# ==============================
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from matplotlib.ticker import MultipleLocator
import numpy as np  # Original numpy
import pennylane as qml
import torch 

# ==============================
# User defined 
# ==============================
import plot_data as pt
import torch_optimizer as tr_opt

# Global Parameters
Tau_global = torch.tensor(0, dtype=torch.float, requires_grad=False)   # Dephase tau
Gamma_ps_global = torch.tensor(0, dtype=torch.float, requires_grad=False)
Paras_global = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.float, requires_grad=True)
Phi_global = torch.tensor(0, dtype=torch.float, requires_grad=True)

def main():
    # ==============================
    # Setup for Quantum Computations
    # ==============================

    # PennyLane settings
    dev = qml.device('default.mixed', wires=3)

    H = qml.Hamiltonian(
        coeffs=[-0.5, -1], 
        observables=[
            qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2), 
            qml.Identity(0) @ qml.PauliZ(1) @ qml.PauliZ(2)
        ]
    )

    H_1 = qml.Hamiltonian(
        coeffs=[-0.5, -0.5, -0.5], 
        observables=[qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2)]
    )

    def Dephase_factor(tau):
        """ 
        Calculate the dephasing factor for a given dephasing time tau.

        Args:
            tau (torch.Tensor): Dephasing time.

        Returns:
            torch.Tensor: Dephasing factor.
        """  
        return 1 - torch.exp(-2 * tau)


    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(phi):
        global Paras_global, Tau_global
        theta_x = Paras_global[0]
        phi_z1 = Paras_global[1]
        phi_z2 = Paras_global[2]
        phi_z3 = Paras_global[3]
        tau_1 = Paras_global[4]
        tau_2 = Paras_global[5]

        gamma_dephase = Dephase_factor(Tau_global)

        # Stage_1: RY for pi/2
        qml.RY(torch.pi/2, wires=0)
        qml.RY(torch.pi/2, wires=1)
        qml.RY(torch.pi/2, wires=2)

        # Stage_2: Entangler    
        qml.ApproxTimeEvolution(H, tau_1, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1)    
        qml.PhaseDamping(gamma_dephase, wires = 2)    

        qml.RX(theta_x, wires = 0)    
        qml.RX(theta_x, wires = 1)    
        qml.RX(theta_x, wires = 2)    

        qml.RY(-torch.pi/2, wires = 0)    
        qml.RY(-torch.pi/2, wires = 1)   
        qml.RY(-torch.pi/2, wires = 2)   

        qml.ApproxTimeEvolution(H, tau_2, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1) 
        qml.PhaseDamping(gamma_dephase, wires = 2) 

        qml.RY(torch.pi/2, wires = 0)    
        qml.RY(torch.pi/2, wires = 1) 
        qml.RY(torch.pi/2, wires = 2) 

        # Stage_3: Accumulator
        qml.ApproxTimeEvolution(H_1, phi, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1) 
        qml.PhaseDamping(gamma_dephase, wires = 2) 

        qml.RZ(phi_z1, wires=0)
        qml.RZ(phi_z2, wires=1)
        qml.RZ(phi_z3, wires=2)

        qml.RX(torch.pi/(2), wires=0)
        qml.RX(torch.pi/(2), wires=1)
        qml.RX(torch.pi/(2), wires=2)

        # return qml.state()
        return qml.density_matrix(wires = [0, 1, 2])


    @qml.qnode(dev, interface = 'torch', diff_method = 'backprop')
    def Post_selection(phi):

        global Paras_global, Gamma_ps_global
        get_density_matrix = circuit(phi)

        # Kraus operator for 8*8 matrix
        K = torch.tensor([
            [torch.sqrt(1 - Gamma_ps_global), 0], 
            [0, 1]
        ], dtype=torch.complex128)

        Numerator = torch.kron(K, torch.kron(K, K)) @ get_density_matrix @ torch.kron(K, torch.kron(K, K)).conj().T
        Denominator = torch.trace(Numerator)

        rho_ps = Numerator / Denominator

        qml.QubitDensityMatrix(rho_ps, wires = [0, 1, 2])

        return qml.density_matrix(wires = [0, 1, 2])  


    def set_circuit(desired_tau_dephase, desired_gamma_post_selection):
        """
        Set the global dephasing rate and post-selection rate for the circuit.

        Args:
            desired_tau_dephase (float): Desired dephasing rate tau.
            desired_gamma_post_selection (float): Desired post-selection rate gamma.
        """
        global Tau_global, Gamma_ps_global 

        Tau_global = torch.tensor(desired_tau_dephase)
        Gamma_ps_global = torch.tensor([desired_gamma_post_selection])


    def cost_function(paras):
        """ 
        Compute the cost using classical Fisher information for the given parameters.

        Args:
            paras (torch.Tensor): Parameters for quantum gates.

        Returns:
            torch.Tensor: Computed cost.
        """
        global Paras_global, Phi_global
        Paras_global = paras

        CFI = qml.qinfo.classical_fisher(Post_selection)(Phi_global)

        return -CFI


    def sweep_cfi(sweep_range, initial_parameters):
        Phi = torch.arange(sweep_range[0], sweep_range[1], sweep_range[2], dtype=torch.float32)
        Data = torch.zeros((len(Phi), 2))
        Data[:,0] = Phi

        global Phi_global
        params_tensor = initial_parameters.clone().requires_grad_(True)

        for phi_idx in range(len(Phi)):
            Phi_global = Phi[phi_idx].clone().requires_grad_(True)

            Data[phi_idx, 1] = -cost_function(params_tensor)

        return Data


    def sweep_by_tau(sweep_range, init_par, tau_dephase, gamma_post_selection):
        for tau_idx, tau_current in enumerate(tau_dephase):
            set_circuit(tau_current, gamma_post_selection)

            temp = sweep_cfi(sweep_range, init_par).detach().numpy()

            if tau_idx == 0:
                Data = np.zeros((len(tau_dephase), len(temp[:,0]), len(temp[0,:])))
                Data[tau_idx][:, :] = temp
            else:
                Data[tau_idx][:, :] = temp

        return Data


    def torch_optimization(sweep_range, initial_parameters, method):
        """ 
        Perform optimization using specified optimizer over a range of phi values.

        Args:
            sweep_range (list): Range of phi values for optimization.
            initial_parameters (torch.Tensor): Initial parameters for optimization.
            method (str): Optimization method ('LBFGS' or 'Adam').

        Returns:
            torch.Tensor: Data tensor containing optimization results.
        """
        Phi = torch.arange(sweep_range[0], sweep_range[1], sweep_range[2], dtype=torch.float32)
        Data = torch.zeros((len(Phi), len(initial_parameters) + 2))
        Data[:,0] = Phi

        global Phi_global
        params_tensor = initial_parameters.clone().requires_grad_(True)

        opt = torch.optim.LBFGS(
                    [params_tensor], 
                    lr=5e-3,              # Learning rate
                    max_iter=80,          # Maximum number of iterations per optimization step
                    max_eval=None,        # Maximum number of function evaluations per optimization step
                    tolerance_grad=1e-12,  # Termination tolerance on the gradient norm
                    tolerance_change=1e-12,# Termination tolerance on the function value/parameter changes
                    history_size=200,      # Update history size
                    line_search_fn='strong_wolfe'  # Using Strong Wolfe line search

            )

        def closure():
            opt.zero_grad()
            loss = cost_function(params_tensor)
            loss.backward()
            return loss

        steps = 10
        f_logs = [cost_function(params_tensor).item()]
        ftol = 1e-9

        # Begin optimization
        for phi_idx in range(len(Phi)):
            Phi_global = Phi[phi_idx].clone().requires_grad_(True)

            # if Phi[phi_idx] < 3:
            #     steps = 15
            # elif Phi[phi_idx] < 10:
            #     steps = 8
            # else:
            #     steps = 8

            for i in range(steps):
                opt.step(closure)

                fval = cost_function(opt.param_groups[0]['params'][0]).item()
                # print(f"{i+1:03d}th iteration, CFI=", fval)
                f_logs.append(fval)
                if np.abs((fval-f_logs[-2])/fval) < ftol:
                    break
                
            formatted_x = [f"{x:.8f}" for x in opt.param_groups[0]['params'][0].detach().numpy()]
            print("CFI =", f"{-fval:.5f}", "Paras =", formatted_x)

            Data[phi_idx, 1] = -fval
            Data[phi_idx, 2:] = opt.param_groups[0]['params'][0]

            # torch.cat(([-fval], opt.param_groups[0]['params'][0].detach().numpy()))

        return Data


    def optimization_by_tau(sweep_range, init_par, tau_dephase, gamma_post_selection, method):
        """ 
        Iterate over different values of tau_dephase and gamma_post_selection for optimization.

        Args:
            sweep_range (list): Range of phi values for optimization.
            init_par (torch.Tensor): Initial parameters for optimization.
            tau_dephase (list): List of dephasing rates tau to iterate over.
            gamma_post_selection (float): Post-selection rate gamma.
            method (str): Optimization method.

        Returns:
            np.ndarray: Numpy array with optimization results for each tau.
        """
        for tau_idx, tau_current in enumerate(tau_dephase):
            set_circuit(tau_current, gamma_post_selection)

            temp = torch_optimization(sweep_range, init_par, method).detach().cpu().numpy()
            if tau_idx == 0:
                Data = np.zeros((len(tau_dephase), len(temp[:,0]), len(temp[0,:])))
                Data[tau_idx][:, :] = temp
            else:
                Data[tau_idx][:, :] = temp

        return Data

    sweep_range = torch.tensor(
        [1e-2, 2*torch.pi, 1e-2], 
        dtype=torch.float, requires_grad=False
    )
    # CFI = 2.29871 Paras = ['0.67683911', '0.50496644', '0.50500232', '0.50496644', '0.31081310', '2.83822560']
    init_par = torch.tensor(
        [0, 0.7, 0.7, 0.7, 0, 0.5], 
        dtype=torch.float
    )

    tau_dephase = 0.1,
    gamma_ps = 0.8

    res = optimization_by_tau(sweep_range, init_par, tau_dephase, gamma_ps, 'LBFGS')

    np.save("result_[0.1]", res)
    # np.savetxt("[0.1].csv", res[0], delimiter=",")

if __name__ == "__main__":
    import sys
    
    sys.stdout = open("log_out", "w")
    sys.stderr = open("log_err", "w")
    
    main()