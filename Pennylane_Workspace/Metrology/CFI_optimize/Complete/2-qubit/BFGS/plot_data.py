from enum import Enum

import numpy as np  # Original numpy
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from matplotlib.ticker import MultipleLocator


class Index(Enum):
    PHI = 0
    CFI = 1
    
    PARAS_START = 2
    THETA_X = 2
    PHI_Z = 3
    
    TAU_1 = 3
    TAU_2 = 4

def plot_result(result_data, tau_dephase, gamma_ps_select, object):
    """ 
    Plot the results of the optimization.

    Args:
        result_data (np.ndarray): Data from optimization.
        tau_dephase (list): List of dephasing rates tau used in optimization.
        object (str): Type of plot ('CFI', 'theta_x', or 'phi_z').

    """
    if object == 'CFI_PS':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.CFI.value], 
                label = f'$\\tau$ = {tau_current}'
            )

        plt.title(f'CFI at $\gamma_{{ps}} = {gamma_ps_select}$')
        plt.xlabel('Time')
        plt.ylabel('CFI')
        plt.grid()
        plt.legend()
        plt.show()
    
    elif object == 'CFI':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.CFI.value], 
                label = f'$\\tau$ = {tau_current}'
            )
            
        plt.title(f'CFI at $\gamma_{{ps}} = {gamma_ps_select}$')
        plt.xlabel('Time')
        plt.ylabel('CFI')
        plt.grid()
        plt.legend()
        plt.show()
        
    elif object == 'theta_x':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.THETA_X.value], 
                label = f'$\\tau$ = {tau_current}'
            )
            
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi], 
            ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        )
        plt.ylim(0, np.pi)

        plt.title(f'Optimized $\\theta_{{x}}$')
        plt.xlabel('Time')
        plt.ylabel('RAD')
        plt.grid()
        plt.legend()
        plt.show()
        
    elif object == 'phi_z':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.PHI_Z.value], 
                label = f'$\\tau$ = {tau_current}'
            )
            
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi], 
            ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        )
        # plt.ylim(-np.pi, 2*np.pi)

        plt.title(f'Optimized $\\phi_{{z}}$')
        plt.xlabel('Time')
        plt.ylabel('RAD')
        plt.grid()
        plt.legend()
        plt.show()
        
    elif object == 'tau_1':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.TAU_1.value], 
                label = f'$\\tau$ = {tau_current}'
            )
            
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi], 
            ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        )
        plt.ylim(0, np.pi)

        plt.title(f'Optimized $\\tau_{1}$')
        plt.xlabel('Time')
        plt.ylabel('RAD')
        plt.grid()
        plt.legend()
        plt.show()
        
    elif object == 'tau_2':
        for tau_idx, tau_current in enumerate(tau_dephase):
            plt.plot(
                result_data[tau_idx][:,Index.PHI.value], 
                result_data[tau_idx][:,Index.TAU_2.value], 
                label = f'$\\tau$ = {tau_current}'
            )
            
        plt.yticks(
            [-np.pi, -np.pi/2, 0, np.pi/2, np.pi, (3*np.pi)/2, 2*np.pi], 
            ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        )
        plt.ylim(0, np.pi)

        plt.title(f'Optimized $\\tau_{2}$')
        plt.title('Optimized CFI')
        plt.xlabel('Time')
        plt.ylabel('RAD')
        plt.grid()
        plt.legend()
        plt.show()