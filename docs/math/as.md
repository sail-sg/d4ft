# Anti-symmetrizer


In this section, we are interested in solving the schrodinger equation for the ground-state e
nergy of a multi-particle system. 

This is equivalent to find the minimum eigenvalue of the hamiltonian operator, and it is also 
equivalent to find the minimum value of the objective function $\min_{\Psi}\langle\Psi|\hat{H}|
\Psi\rangle$.


we introduce the problem as a optimization objective and constraints.
Define permutation operator as $(P_{ij}\Psi)(r_1,\cdots,r_i,\cdots,r_j,\cdots)=\Psi(r_1,\cdots,r_j,\cdots,r_i,\cdots)$.

Searching for ground-state energy of a system of $n$ fermions, we have the following optimization problem:

$$
\begin{align}
&\min_{\Psi}\langle\Psi|\hat{H}|\Psi\rangle\\
\text{s.t.}\;&\langle\Psi|\Psi\rangle=1\\
&P_{ij}\Psi=-\Psi;\;\forall{i,j}
\end{align}
$$

**Objective**: as mentioned above, the hamiltonian consists of multiple terms. The objective we optimize is the overall energy of the system. We are finding the minimum energy, ground-state enregy, of the system.

**Constraints**: There're two constraints:

- the square of the wave function has to be a probability density function. 
    
- The wave function of fermions needs to be anti-symmetric (Pauli exclusion principle).


To solve the above optimization problem, we need to optimize the function $\Psi$. From deep learning point of view, we can use a neural network $\hat{\Psi}_\theta(\boldsymbol{r})$ parametrized by $\theta$. However, don't forget that $\Psi$ need to satisfy the antisymmetry constraint, which the neural network doesn't satisfy.

A general way to construct anti-symmetric wave-functions is through the antisymmetrizer $\mathcal{A}$. Which is defined as 

$$
\Psi_\theta=\mathcal{A}\hat{\Psi}_\theta=\sum_{\pi\in\Pi}(-1)^{\mathrm{inv}(\pi)}\hat{\Psi}_\theta(r_{\pi(1)},r_{\pi(2)},\cdots,r_{\pi(n)})
$$

Therefore, for any function $\hat{\Psi}_\theta$ that is not anti-symmetric, $\Psi_\theta=\mathcal{A}\hat{\Psi}_\theta$ is an anti-symmetric function. Notice that $\Pi$ contains $n!$ permutations, therefore, the calculation of the above $\mathcal{A}\hat{\Psi}_\theta$ is exponential in $n$. 
