# Quantum_Optimization WorkSpace


### PennyLane Settings
To begin, we configure PennyLane to use a specific quantum device. In this case, we are using PennyLane's default mixed-state simulator.

##### Parameters:
- `default.mixed`: The 'default.mixed' device is capable of simulating quantum states that are mixed

- `wires=n`: Defines the number of n_qubits

### Hamiltonian Definition

#### [1-qubit]
$$H = -0.5 \cdot \sigma_z$$

$$where,
\sigma_z = \begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}$$

#### [2-qubit]
#### Entangler Hamiltonian
$$H = -0.5 \cdot Z_0 \otimes Z_1$$

$$where,Z = \begin{pmatrix} 
1 & 0 \\ 
0 & -1 \end{pmatrix}$$

#### Phase Accumulator Hamiltonian
$$H_{1} = -0.5 \cdot (Z_0 + Z_1) $$

$$= -0.5 \cdot (Z \otimes I) - 0.5 \cdot (I \otimes Z)$$


$$where,
Z = \begin{pmatrix} 
1 & 0 \\ 
0 & -1 \end{pmatrix}$$

#### [3-qubit]
#### Entangler Hamiltonian
$$H = -0.5 \cdot Z_0 \otimes Z_1 \otimes I_{2} 
-0.5 \cdot I_{0} \otimes Z_1 \otimes Z_2$$

$$where,
Z = \begin{pmatrix} 
1 & 0 \\ 
0 & -1 \end{pmatrix}$$

#### Phase Accumulator Hamiltonian
$$H_{1} = -0.5 \cdot (Z_0 + Z_1 + Z_2)$$

$$where, Z = \begin{pmatrix} 
1 & 0 \\
0 & -1 \end{pmatrix}$$

#### [4-qubit]
#### Entangler Hamiltonian
$$H = -0.5 \cdot(Z_0 \otimes Z_1 \otimes I_{2} \otimes I_3 + Z_{0} \otimes I_1 \otimes Z_{2} \otimes I_3 + Z_{0} \otimes I_1 \otimes I_{2} \otimes Z_3 + $$

$$I_{0} \otimes Z_1 \otimes Z_{2} \otimes I_{3} + I_{0} \otimes Z_1 \otimes I_{2} \otimes Z_{3} + I_{0} \otimes I_1 \otimes Z_{2} \otimes Z_3)$$

$$where, Z = \begin{pmatrix} 
1 & 0 \\ 
0 & -1 
\end{pmatrix}$$

#### Phase Accumulator Hamiltonian
$$H_{1} = -0.5 \cdot (
    Z_0 + Z_1 + Z_2 + Z_3
) $$

$$where, Z = \begin{pmatrix} 
1 & 0 \\ 
0 & -1 \end{pmatrix}$$


##### Parameters:
- `coeffs`: A list of coefficients for each term in the Hamiltonian.

- `observables`: A list of quantum observables that constitute the Hamiltonian.

#### Reference
https://docs.pennylane.ai/en/stable/code/api/pennylane.ApproxTimeEvolution.html
https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliRot.html

### Dephasing Factor Calculation
The function Dephase_factor(tau) is designed to calculate the dephasing factor

##### Parameters:
- `tau (torch.Tensor)`: The time scale over which a quantum system loses its phase coherence.


### Time-evolution and Dephasing Density Matrix
$$Let, e^{-t/T_2} = e^{-\tau}$$
Here, t is the actual time, and T2 is the dephasing time constant. The term τ represents the normalized dephasing time.

The effect of dephasing on a single qubit in the computational basis, considering a phase ϕ, can be described by the following density matrix:

$$\frac{1}{2} 
\begin{bmatrix} 
1 & e^{(i\phi - \tau)} \\ 
e^{(-i\phi - \tau)} & 1
\end{bmatrix}$$

### Time-evolution with Phase Damping
Phase damping, as implemented in PennyLane, can be described by a similar density matrix:

$$\frac{1}{2} 
\begin{bmatrix}
1 & e^{i\phi} \sqrt{1 - \gamma} \\
e^{-i\phi} \sqrt{1 - \gamma} & 1
\end{bmatrix}$$


### Relating Dephasing and Phase Damping Parameters

$$\gamma = 1 - e^{-2 \tau}$$

This equation provides a direct way to calculate the phase damping parameter γ from the dephasing time 
τ. We can directly use this value as an argument for the phase damping channel in PennyLane.

Furthermore,
$$e^{-\tau} = \sqrt{1 - \gamma}$$

### Usage in PennyLane
For more information on PennyLane's implementation of the phase damping channel, you can refer to the documentation at:
https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseDamping.html

### PennyLane QNode Decoration:
- `@qml.qnode(dev, interface='torch', diff_method='backprop')`

#### Parameters:
- `dev`: The quantum device that executes the computations.
- `interface='torch'`: This setting ensures that the QNode is compatible with PyTorch, allowing for automatic differentiation.
- `diff_method='backprop'`: Specifies that backpropagation is used for computing gradients within the PyTorch framework.


### Post-selection for Single-qubit Circuit 
The `Post_selection(phi)` function encapsulates the process of post-selection within a quantum circuit.

### Implementation details

- Quantum Circuit Execution
: We call the function `circuit(phi)`, which returns the density matrix of the system as a complex tensor in PyTorch.

- Kraus operator 
: A Kraus operator `K` is defined to model the noise process or selective measurements

$$K = \begin{bmatrix}
\sqrt{1 - \gamma_{ps}} & 0 \\
0 & 1
\end{bmatrix}$$

- Application of Post-selection 
: The post-selection process is modeled by applying the Kraus operator `K` to the system's density matrix `ρ`, and normalizing the result by the trace of the numerator to obtain the post-selected density matrix 
ρ_ps

#### [1-qubit]
$$\rho_{\text{ps}} = \frac{K \rho K^\dagger}{\text{Tr}{[K \rho K^\dagger]}}$$

#### [2-qubit]
$$\rho_{ps} = \frac{(K \otimes I) \rho (K \otimes I)^{\dagger}}{Tr[(K \otimes I) \rho (K \otimes I)^{\dagger}]}$$

#### [3-qubit]
$$\rho_{ps} = 
\frac{(K \otimes(K \otimes K)) \rho (K \otimes(K \otimes K))^{\dagger}}
{Tr[(K \otimes(K \otimes K)) \rho (K \otimes(K \otimes K))^{\dagger}]}$$

#### [4-qubit]
$$\rho_{ps} = \frac
{(K \otimes K_3) \rho (K \otimes K_3)^{\dagger}}
{Tr[(K \otimes K_3) \rho (K \otimes K_3)^{\dagger}]}$$

$$ where, K = \begin{bmatrix}
\sqrt{1-\gamma_{ps}} & 0 \\
0 & 1 
\end{bmatrix}  
, K_2 =  (K \otimes K) and K_3 =  (K \otimes K_2)$$

### Cost-function to Maximize
The function cost_function(paras) is specifically designed to compute the Classical Fisher Information (CFI) with respect to a set of parameters. The CFI is an important quantity in quantum parameter estimation and is given by the formula:

$$\text{CFI}(\theta) = \sum_{x} \frac{1}{P(x|\theta)} \left( \frac{\partial P(x|\theta)}{\partial \theta} \right)^2$$

#### `qml.qinfo.classical_fisher`
The function `qml.qinfo.classical_fisher` within the PennyLane library computes the CFI for the supplied parameters. By invoking this function with a quantum circuit's parameters, it yields a matrix whose size is proportional to the number of parameters involved. However, in our specific application, we focus on the time-evolution parameter `phi_global` instead of local variables. This strategic choice is driven by the goal to assess the CFI with respect to time-evolution only.

Using global variables like `phi_global` allows us to selectively compute the CFI for this singular, crucial parameter, bypassing the necessity to evaluate a potentially large parameter space. This approach streamlines the computation, focusing our resources and optimization efforts on the time-evolution aspect.

To gain a deeper understanding of how `qml.qinfo.classical_fisher` is implemented and its functionalities, you can refer to the official PennyLane documentation at:
https://docs.pennylane.ai/en/latest/code/api/pennylane.qinfo.transforms.classical\_fisher.html

#### (-) sign
The purpose is to amplify the CFI, thereby enhancing the precision of parameter estimation. Given that PyTorch optimizers inherently minimize cost functions, the CFI's sign is inverted in the return statement of our cost_function.


### `torch_optimization`

#### Purpose
The `torch_optimization` function performs optimization over a range of Phi values using the specified optimization method ('LBFGS' or 'Adam'). The function iteratively optimizes the quantum gate parameters, aiming to minimize a predefined cost function.

#### Arguments

- `sweep_range` (list): A list specifying the start, end, and step values to generate a range of ϕ values for optimization.
- `initial_parameters` (torch.Tensor): A tensor containing the initial guess for the parameters that will be optimized.
- `method` (str): A string indicating the optimization algorithm to be used ('LBFGS' or 'Adam').

#### Returns:
- `torch.Tensor`: A 2-dimensional tensor containing the optimized results. The first column holds the time-evolution values from `sweep_range`, the second column stores - the optimized parameter θ_x, and the third column contains the optimized parameter ϕ_z.

#### Data structure
The function generates a 2-dimensional torch tensor Data to store the results:
- `Data[:,0]`: stores the Phi values (time of time-evolution) obtained from `sweep_range`.
- `Data[:,1]`: stores the optimized value of the cost function (e.g., negative Classical Fisher Information (CFI)).
- `Data[:,2:]`: stores the optimized parameters of the quantum gate (`theta_x` and `phi_z`).

#### Execution process
1. The function computes a tensor Phi to hold the range of phase values.

2. A zero-initialized tensor Data is prepared to store the optimization results. 

3. For each value of ϕ in Phi, the function performs the optimization using the provided method, updating the global variable `Phi_global`.

4. The optimized parameters and corresponding cost function values are recorded in Data.


### `select_optimizer`

#### Purpose
The select_optimizer function selects and configures a PyTorch optimizer based on the specified method ('LBFGS' or 'Adam').

#### Arguments
- `method`: 
- `parameters_in`:

#### Available Optimizers
- LBFGS: This is an optimization algorithm in the family of quasi-Newton methods. It approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using a limited amount of computer memory. 

https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html

- Adam: A stochastic optimization method that computes adaptive learning rates for each parameter.

https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
