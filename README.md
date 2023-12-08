# Quantum_Optimization Work-Space

### Dephasing Factor Calculation
The function Dephase_factor(tau) is designed to calculate the dephasing factor

##### Parameters:
- `tau (torch.Tensor)`: The time scale over which a quantum system loses its phase coherence.


### Time-evolution and Dephasing Density Matrix
$$Let, e^{-t/T_2} = e^{-\tau}$$
Here, t is the actual time, and T2 is the dephasing time constant. The term τ represents the normalized dephasing time.

The effect of dephasing on a single qubit in the computational basis, considering a phase ϕ, can be described by the following density matrix:

$$\frac{1}{2} \begin{bmatrix}1 & e^{(i\phi - \tau)} \\e^{(-i\phi - \tau)} & 1\end{bmatrix}$$

### Time-evolution with Phase Damping
Phase damping, as implemented in PennyLane, can be described by a similar density matrix:

$\frac{1}{2} 

\begin{bmatrix}

1 & e^{i\phi} \sqrt{1 - \gamma} \\
e^{-i\phi} \sqrt{1 - \gamma} & 1

\end{bmatrix}$


### Relating Dephasing and Phase Damping Parameters

$$\gamma = 1 - e^{-2 \tau}$$

This equation provides a direct way to calculate the phase damping parameter γ from the dephasing time 
τ. We can directly use this value as an argument for the phase damping channel in PennyLane.

Furthermore,
$$ e^{-\tau} = \sqrt{1 - \gamma}$$

### Usage in PennyLane
For more information on PennyLane's implementation of the phase damping channel, you can refer to the documentation at:
https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseDamping.html
