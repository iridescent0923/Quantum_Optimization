# ==============================
# Third-party Library Imports
# ==============================
import matplotlib.pyplot as plt
from IPython.display import display, Latex
from matplotlib.ticker import MultipleLocator
import numpy as np  # Original numpy
import pennylane as qml
import scipy as sp
from autograd import grad, jacobian

# Pennylane numpy
from pennylane import numpy as pnp 

def fisher_information(phi, type, circuit):
    phi_current = pnp.array([phi], requires_grad = True)
    
    if type == 'classical':
            return qml.qinfo.classical_fisher(circuit)(phi_current[0])
        
    else:
            return qml.qinfo.quantum_fisher(circuit)(phi_current[0]) 
        
def plot_fisher(type, num_qubit, circuit):
    PHI = np.arange(1e-2, 3*np.pi + 1e-2, 1e-2)
    Data = np.zeros((len(PHI), 2))
    Data[:,0] = PHI.squeeze()

    for phi_idx, phi in enumerate(PHI):
        Data[phi_idx, 1] = fisher_information(phi, type, circuit)

    plt.plot(Data[:, 0], Data[:,1], label = f'{num_qubit}-qubit')
    plt.title(f'Bell state')
    plt.xlabel('Time')
    if type == 'classical':
        plt.ylabel('CFI')
    else:
        plt.ylabel('QFI')

    plt.grid()
    plt.legend()