# Quantum_Optimization Work-Space

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

$$\rho_{\text{ps}} = \frac{K \rho K^\dagger}{\text{Tr}{[K \rho K^\dagger]}}$$
