import numpy as np

def get_constraints(phi_current, gamma_ps, tau_current):
    """
    Calculate the constraints for the optimization based on current phi, gamma and tau values.

    Args:
        phi_current (float): The current value of phi in the optimization loop.
        gamma_ps (float): Gamma value for post-selection.
        tau_current (float): Current value of tau.

    Returns:
        list of tuple: Constraints for the optimization variables.
    """
    
    N = 2*np.pi * int(phi_current / (2*np.pi))
    if gamma_ps == 8e-1:
        if tau_current == 0:
            # return [(-float('inf'), float('inf'))] * 2
            return [
                (np.pi/2, np.pi/2),
                (-np.pi/2, 3*np.pi/2)
            ]

        elif tau_current == (5e-2):
            if phi_current < 0.45 + N:
                return [(np.pi/2, np.pi)] * 2
            elif phi_current <= 1.02 + N:
                return [(np.pi/2, 3.70)] * 2
            elif phi_current <= 1.57 + N:
                return [(np.pi/2, 4.24006637)] * 2
            elif 3.00 + N <= phi_current <= 3.67 + N:
                return [(np.pi/2, np.pi/2), (0.32993364, 0.99993333)]
            elif 3.69 + N <= phi_current <= 4.0 + N:
                return [(np.pi/2, np.pi/2), (1.01993369, 1.32993365)]
            elif 4.03 + N <= phi_current <= 4.22 + N:
                return [(np.pi/2, np.pi/2), (1.35993364, 1.54993374)]
            elif 4.24 + N <= phi_current <= 4.69 + N:
                return [(np.pi/2, np.pi/2), (1.56993364, 2.01993363)]
            elif (4.82) + N <= phi_current <= (5.5) + N:
                return [(np.pi/2, np.pi/2), (1.20688106, 1.88688109)]
            elif 5.5 + N <= phi_current <= (6.0) + N:
                return [(np.pi/2, np.pi/2), (1.88688109, 2.38688106)]
            elif 6.05 + N <= phi_current <= (6.15) + N:
                return [(np.pi/2, np.pi/2), (2.43688084, 2.53688103)]

        elif tau_current == 2e-1:
            if phi_current <= 0.5 + N:
                return [(0, np.pi)] * 2
            elif phi_current < 1.58 + N:
                return [(-0.8439553621272445, 3.9939553536152146)] * 2 
            elif phi_current < 2.42 + N:
                return [(-0.8439553621272445, 3.9939553536152146)] * 2 
            elif phi_current < 4 + N:
                return [(0, np.pi/2)] * 2 
            elif phi_current < (1.57 + 3.14) + N:
                return [(np.pi/2, np.pi)] * 2 
        
        elif tau_current == 5e-1:
            return [(-5e-1, np.pi + 6e-1 )] * 2
        
        elif 1 <= tau_current <= 4:
            return [(-5e-1, np.pi + 35e-2)] * 2