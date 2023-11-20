import torch

def select_optimizer(method, parameters_in):
    """
    Select and configure an optimizer based on the specified method.

    Args:
        method (str): The optimization method to use ('LBFGS' or 'Adam').
        parameters_in (torch.Tensor): The parameters to be optimized.

    Returns:
        torch.optim.Optimizer: Configured optimizer object.

    Raises:
        ValueError: If an invalid optimization method is specified.

    This function initializes and returns a PyTorch optimizer based on the provided method.
    For 'LBFGS', it sets up an LBFGS optimizer with specified learning rate, maximum iterations,
    tolerance levels, and history size. For 'Adam', it sets up an Adam optimizer with specified
    learning rate, beta values, epsilon for numerical stability, and weight decay options.
    If a method other than 'LBFGS' or 'Adam' is provided, the function raises a ValueError.
    """
    
    if method == 'LBFGS':
        opt = torch.optim.LBFGS(
                [parameters_in], 
                lr=0.01,              # Learning rate
                max_iter=20,          # Maximum number of iterations per optimization step
                max_eval=None,        # Maximum number of function evaluations per optimization step
                tolerance_grad=1e-7,  # Termination tolerance on the gradient norm
                tolerance_change=1e-9,# Termination tolerance on the function value/parameter changes
                history_size=100      # Update history size
        )
        return opt
    
    elif method == 'Adam':
        opt = torch.optim.Adam(
            [parameters_in],
            lr=0.001,                # Learning rate (default: 0.001)
            betas=(0.9, 0.999),      # Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            eps=1e-08,               # Term added to the denominator to improve numerical stability (default: 1e-8)
            weight_decay=0,          # Weight decay (L2 penalty) (default: 0)
            amsgrad=False            # Whether to use the AMSGrad variant of this algorithm (default: False)
        )
        return opt
    
    else:
        raise ValueError("Invalid optimizer choice.")