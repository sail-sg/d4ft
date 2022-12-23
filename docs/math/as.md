# Anti-symmetrizer


In this section, we introduce the problem as a optimization objective and constraints.
Define permutation operator as $(P_{ij}\Psi)(r_1,\cdots,r_i,\cdots,r_j,\cdots)=\Psi(r_1,\cdots,r_j,\cdots,r_i,\cdots)$

$$
\begin{align}
&\min_{\Psi}\langle\Psi|\hat{H}|\Psi\rangle\\
s.t.\;&\langle\Psi|\Psi\rangle=1\\
&P_{ij}\Psi=-\Psi;\;\forall{i,j}
\end{align}
$$

**Objective**: as mentioned above, the hamiltonian consists of multiple terms. The objective we optimize is the overall energy of the system.
**Constraints**: There're two constraints, 1. the square of the wave function has to be a probability density function. 2. The wave function of fermions needs to be anti-symmetric (Pauli exclusion principle).

<!-- ### Anti-symmetry Constraint -->

To solve the above optimization problem, we need to optimize the function $\Psi$. From deep learning point of view, we can use a neural network $\hat{\Psi}_\theta(\boldsymbol{r})$. However, don't forget that $\Psi$ need to satisfy the antisymmetry constraint, which the neural network doesn't satisfy.

A general way to construct anti-symmetric wave-functions is through the antisymmetrizer $\mathcal{A}$. Which is defined as 

$$
\Psi_\theta=\mathcal{A}\hat{\Psi}_\theta=\sum_{\pi\in\Pi}(-1)^{\mathrm{inv}(\pi)}\hat{\Psi}_\theta(r_{\pi(1)},r_{\pi(2)},\cdots,r_{\pi(n)})
$$

Therefore, for any function $\hat{\Psi}_\theta$ that is not anti-symmetric, $\Psi_\theta=\mathcal{A}\hat{\Psi}_\theta$ is an anti-symmetric function. Notice that $\Pi$ contains $n!$ permutations, therefore, the calculation of the above $\mathcal{A}\hat{\Psi}_\theta$ is exponential in $n$. 