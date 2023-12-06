#!/bin/env/python3
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


# ==============================
# Setup for Quantum Computations
# ==============================
# Global Parameters
Tau_global = torch.tensor(0, dtype=torch.float, requires_grad=False)   # Dephase tau
Gamma_ps_global = torch.tensor(0, dtype=torch.float, requires_grad=False)
Paras_global = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, requires_grad=True)
Phi_global = torch.tensor(0, dtype=torch.float, requires_grad=True)

def main():
    # PennyLane settings
    dev = qml.device('default.mixed', wires=4)
    
    # Define Hamiltonian for quantum computations
    H = qml.Hamiltonian(
        coeffs=[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], 
        observables=[
            qml.PauliZ(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.Identity(3),
            qml.PauliZ(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.Identity(3),
            qml.PauliZ(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliZ(3),
            qml.Identity(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.PauliZ(3),
            qml.Identity(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
        ]
    )
    
    H_1 = qml.Hamiltonian(
        coeffs=[-0.5, -0.5, -0.5, -0.5], 
        observables=[
            qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)
        ]
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
    
    
    
    class INDEX(Enum):
        THETA_X1 = 0
        THETA_X2 = 1
        
        PHI_Z1 = 2
        PHI_Z2 = 3
        PHI_Z3 = 4
        PHI_Z4 = 5
        
        TAU_L1 = 6
        TAU_L2 = 7
        
        TAU_R1 = 8
        TAU_R2 = 9
    
    
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(phi):
        global Paras_global, Tau_global
        theta_x1 = Paras_global[0]
        theta_x2 = Paras_global[1]
        
        phi_z1 = Paras_global[2]
        phi_z2 = Paras_global[3]
        phi_z3 = Paras_global[4]
        phi_z4 = Paras_global[5]
        
        tau_L1 = Paras_global[6]
        tau_L2 = Paras_global[7]
        
        tau_R1 = Paras_global[8]
        tau_R2 = Paras_global[9]
        
        gamma_dephase = Dephase_factor(Tau_global)
        
        # Stage_1: RY for pi/2
        qml.RY(torch.pi/2, wires=0)
        qml.RY(torch.pi/2, wires=1)
        qml.RY(torch.pi/2, wires=2)
        qml.RY(torch.pi/2, wires=3)
        
        # Stage_2: Entangler_layer_1    
        qml.ApproxTimeEvolution(H, tau_L1, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1)    
        qml.PhaseDamping(gamma_dephase, wires = 2)    
        qml.PhaseDamping(gamma_dephase, wires = 3)    
        
        qml.RX(theta_x1, wires = 0)    
        qml.RX(theta_x1, wires = 1)    
        qml.RX(theta_x1, wires = 2)    
        qml.RX(theta_x1, wires = 3)    
    
        qml.RY(-torch.pi/2, wires = 0)    
        qml.RY(-torch.pi/2, wires = 1)   
        qml.RY(-torch.pi/2, wires = 2)   
        qml.RY(-torch.pi/2, wires = 3)   
    
        qml.ApproxTimeEvolution(H, tau_R1, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1) 
        qml.PhaseDamping(gamma_dephase, wires = 2) 
        qml.PhaseDamping(gamma_dephase, wires = 3) 
        
        qml.RY(torch.pi/2, wires = 0)    
        qml.RY(torch.pi/2, wires = 1) 
        qml.RY(torch.pi/2, wires = 2) 
        qml.RY(torch.pi/2, wires = 3) 
        
        # Stage_3: Entangler_layer_2    
        qml.ApproxTimeEvolution(H, tau_L2, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1)    
        qml.PhaseDamping(gamma_dephase, wires = 2)    
        qml.PhaseDamping(gamma_dephase, wires = 3)    
        
        qml.RX(theta_x2, wires = 0)    
        qml.RX(theta_x2, wires = 1)    
        qml.RX(theta_x2, wires = 2)    
        qml.RX(theta_x2, wires = 3)    
    
        qml.RY(-torch.pi/2, wires = 0)    
        qml.RY(-torch.pi/2, wires = 1)   
        qml.RY(-torch.pi/2, wires = 2)   
        qml.RY(-torch.pi/2, wires = 3)   
    
        qml.ApproxTimeEvolution(H, tau_R2, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1) 
        qml.PhaseDamping(gamma_dephase, wires = 2) 
        qml.PhaseDamping(gamma_dephase, wires = 3) 
        
        qml.RY(torch.pi/2, wires = 0)    
        qml.RY(torch.pi/2, wires = 1) 
        qml.RY(torch.pi/2, wires = 2) 
        qml.RY(torch.pi/2, wires = 3) 
        
        # Stage_4: Accumulator
        qml.ApproxTimeEvolution(H_1, phi, 1)
        qml.PhaseDamping(gamma_dephase, wires = 0)
        qml.PhaseDamping(gamma_dephase, wires = 1) 
        qml.PhaseDamping(gamma_dephase, wires = 2) 
        qml.PhaseDamping(gamma_dephase, wires = 3) 
        
        qml.RZ(phi_z1, wires=0)
        qml.RZ(phi_z2, wires=1)
        qml.RZ(phi_z3, wires=2)
        qml.RZ(phi_z4, wires=3)
        
        qml.RX(torch.pi/(2), wires=0)
        qml.RX(torch.pi/(2), wires=1)
        qml.RX(torch.pi/(2), wires=2)
        qml.RX(torch.pi/(2), wires=3)
        
        # return qml.state()
        return qml.density_matrix(wires = [0, 1, 2, 3])
    
    
    @qml.qnode(dev, interface = 'torch', diff_method = 'backprop')
    def Post_selection(phi):
    
        global Paras_global, Gamma_ps_global
        get_density_matrix = circuit(phi)   # 16*16 for 4-qubit
    
        K = torch.tensor([
            [torch.sqrt(1 - Gamma_ps_global), 0], 
            [0, 1]
        ], dtype=torch.complex128)  # 2*2
        K_2 = torch.kron(K, K)      # 4*4
        K_3 = torch.kron(K, K_2)    # 8*8
        
        I_2 = torch.eye(2)            # 2*2
        
        # num_1 = torch.kron(K_3, I_2) @ get_density_matrix @ torch.kron(K_3, I_2).conj().T
        # num_2 = torch.kron(I_2, K_3) @ get_density_matrix @ torch.kron(I_2, K_3).conj().T
        
        # den_1 = 2*torch.trace(num_1)
        # den_2 = 2*torch.trace(num_2)
        
        # rho_1 = num_1 / den_1
        # rho_2 = num_2 / den_2
        
        # rho_ps = rho_1 + rho_2
        
        num = torch.kron(K, K_3) @ get_density_matrix @ torch.kron(K, K_3).conj().T
        den = torch.trace(num)
        
        rho_ps = num / den
            
        qml.QubitDensityMatrix(rho_ps, wires = [0, 1, 2, 3])
        
        return qml.density_matrix(wires = [0, 1, 2, 3])  
    
    
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
        
        opt = tr_opt.select_optimizer(method, params_tensor)
        
        def closure():
            opt.zero_grad()
            loss = cost_function(params_tensor)
            loss.backward()
            return loss
           
        steps = 20
        f_logs = [cost_function(params_tensor).item()]
        ftol = 1e-10
            
        # Begin optimization
        for phi_idx in range(len(Phi)):
            Phi_global = Phi[phi_idx].clone().requires_grad_(True)
    
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
        [1e-2, 2*torch.pi, 1e-3], 
        dtype=torch.float, requires_grad=False
    )
    
    init_par = torch.tensor([
        # theta_x1:x2
        torch.pi/4, torch.pi/4,
        
        # phi_z1:z4
        torch.pi/4, torch.pi/4, torch.pi/4, torch.pi/4, 
        
        # tau_L1:L2
        torch.pi/4,torch.pi/4,
        
        # tau_R1:R2
        torch.pi/4,torch.pi/4
        ], dtype=torch.float)
    
    
    tau_dephase = 0,
    gamma_ps = 0.8
    
    res = optimization_by_tau(sweep_range, init_par, tau_dephase, gamma_ps, 'LBFGS')
    
    
    np.save(f"result_[{tau_dephase[0]}]", res)

if __name__ == "__main__":
    import sys
    
    sys.stdout = open("log_out[0]", "w")
    sys.stdout = open("log_err[0]", "w")
    
    main()