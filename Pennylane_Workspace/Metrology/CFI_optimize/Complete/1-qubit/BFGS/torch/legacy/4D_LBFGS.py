# Data Index
class Layer(Enum):
    BEFORE_OPT = 0
    AFTER_OPT = 1
    NUMBER = 2

class Column(Enum):
    PHI = 0
    CFI = 1
    PARAS_BEGIN = 2
    THETA_X = 2
    PHI_Z = 3

class Index:
    layer = Layer
    column = Column
    
# == BFGS -> Return Data_set:[phi, CFI, 6-Paras] ==
def run_optimization(sweep_range, initial_parameters, gamma_ps, tau_dephase, circuit_select):
    
    # Create Data array
    PHI = np.arange(sweep_range[0], sweep_range[1], sweep_range[2])    
    Data = np.zeros((
        len(tau_dephase), 
        Index.layer.NUMBER.value, 
        len(PHI), 
        len(initial_parameters) + Index.column.PARAS_BEGIN.value
    )) 
    Data[:, :, :, Index.column.PHI.value] = PHI.squeeze() # Append PHI in to 0th col
    
    # Set global variables
    global Gamma_ps_global, Phi_global, Paras_global, Tau_global, Circuit_select_global
    Circuit_select_global = circuit_select
    Gamma_ps_global = gamma_ps 
        
    for tau_idx, tau_current in enumerate(tau_dephase):
        Tau_global = tau_current
        
        num_layer = int(Index.layer.NUMBER.value)
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                pass
                # for phi_idx in range(len(PHI)):
                #     Phi_global = phi_current
                    
                #     paras_to_tensor = torch.tensor(initial_parameters, requires_grad=True)
                #     Data[tau_idx][Index.layer.BEFORE_OPT.value][phi_idx][Index.column.CFI.value] = -Cost_function(paras_to_tensor).detach()[0].numpy()[0]
                #     Data[tau_idx][Index.layer.BEFORE_OPT.value][phi_idx][Index.column.PARAS_BEGIN.value:] = initial_parameters
            
            else:
                for phi_idx, phi_current in enumerate(PHI):
                    Phi_global = torch.tensor(phi_current, dtype=torch.float)
                    # Constraints = get_constraints(phi_current, Gamma_ps_global, len(initial_parameters))
                    
                    Result_BFGS = BFGS(initial_parameters)
                    Data[tau_idx][Index.layer.AFTER_OPT.value][phi_idx][Index.column.CFI.value] = -Result_BFGS[0]
                    Data[tau_idx][Index.layer.AFTER_OPT.value][phi_idx][Index.column.PARAS_BEGIN.value:] = -Result_BFGS[1:]
    
    return Data

def BFGS(initial_parameters):
    
    params_t = torch.tensor(initial_parameters, requires_grad=True)
    # params_t = torch.tensor(initial_parameters, dtype=torch.float).requires_grad_(True)

    opt = torch.optim.LBFGS(
    [params_t], 
    lr=0.01,              # Learning rate
    max_iter=20,          # Maximum number of iterations per optimization step
    max_eval=None,        # Maximum number of function evaluations per optimization step
    tolerance_grad=1e-7,  # Termination tolerance on the gradient norm
    tolerance_change=1e-9,# Termination tolerance on the function value/parameter changes
    history_size=100      # Update history size
    )
    

    steps = 500           # Maximum steps

    f_logs = [Cost_function(params_t).item()]
    ftol = 1e-10

    def closure():
        opt.zero_grad()
        loss = Cost_function(params_t)
        loss.backward()
        return loss

    for i in range(steps):
        opt.step(closure)
        fval = Cost_function(opt.param_groups[0]['params'][0]).item()
        # print(f"{i+1:03d}th iteration, CFI=", fval)
        f_logs.append(fval)
        if np.abs((fval-f_logs[-2])/fval) < ftol:
            print("CFI=", -fval, "Paras=", opt.param_groups[0]['params'][0].detach().numpy())
            break
        
    result_optimization = np.zeros(len(initial_parameters) + 1)
    result_optimization[Index.column.CFI.value - 1] = fval
    result_optimization[Index.column.PARAS_BEGIN.value - 1:] = opt.param_groups[0]['params'][0].detach().numpy()
    
    return result_optimization


def get_constraints(phi_current, gamma_ps, len_param):
    """
    Calculate the constraints for the optimization based on current phi, gamma and tau values.

    Args:
        phi_current (float): The current value of phi in the optimization loop.
        Gamma_ps_global (float): Gamma value for post-selection.
        tau_current (float): Current value of tau.

    Returns:
        list of tuple: Constraints for the optimization variables.
    """
    
    N = 2*np.pi * int(phi_current / (2*np.pi))
    return [(-float('inf'), float('inf'))] * len_param
