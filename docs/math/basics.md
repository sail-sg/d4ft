# Schrodinger Equation

Time-independent Schrödinger equation can be defined by:

$$
\hat H | \boldsymbol{\Psi} \rangle = \varepsilon | \boldsymbol{\Psi} \rangle
$$

where $\hat H$ is the hamiltonian operator and $\varepsilon$ is the associate energy. 

Solving the above equation is equivalent to solving the eigenvalue/eigenvector problem for the 
hamiltonian operator. Among the eigenvalues, the minimum one is the ground-state energy of the 
system.

For different systems, the form of the hamiltonian operator is different, we will see some 
examples below.

>**Example** (Hamiltonian for Hydrogen atom).
>As there is only one electron in a Hydrogen atom, there are three components of the hamiltonian: nuclear kinetic energy,  electronic kinetic energy and the electron-nucleus attraction. Due to the Born-Oppenheimer approximation, the nucleus is much larger and is assumed frozen, the hamiltonian can be simplified as,

> $$\hat H = -\dfrac{1}{2}\nabla^2 - \dfrac{1}{r-R},$$

> where $\nabla^2$ is the Laplacian operator for the kinetic energy, and the second term is the Coulomb potential operator with $r$ and $R$ denote the location of electron and nucleus respectively.

>**Example** (Hamiltonian for water molecule).
>\begin{equation}
\hat H = \underbrace{-\sum_{i=1}^{10}\dfrac{1}{2}\nabla^2_{r_i}}_{\text{Kinetic energy} \\ \text{of electron i}} - \underbrace{\sum_{i=1}^{10} \dfrac{8}{|r_i - R_{O}|}}_{\text{Electron attraction} \\ \text{to the Oxygen atom}}  - \underbrace{\sum_{k=1}^2\sum_{i=1}^{10} \dfrac{1}{r_i-R_{H_k}} }_{\text{Electron attraction to} \\ \text{two Hydrogen atoms}} + \underbrace{ \sum_{i=1}^{10} \sum_{j=1}^{10} \dfrac{1}{\vert r_i-r_j \vert}}_{\text{Electron-electron} \\ \text{repulsion}} + \underbrace{ \sum_{i=1}^{3} \sum_{j=1}^{3} \dfrac{e_ie_j}{\vert R_i-R_j \vert}}_{\text{Nucleus-nucleus} \\ \text{repulsion}} 
\end{equation}


### Wave function for multi-particle system
A wave function in quantum physics is a mathematical description of the quantum state of an isolated quantum system. The wave function is a complex-valued probability amplitude, and the probabilities for the possible results of measurements made on the system can be derived from it.

Wave function tells the probability that a particle will be in the interval $a<r<b$:

$$
P\left( a<r<b \right) = \int_a^b | \Psi(r) |^2 dr.
$$

For an $N$-particle wave funtion $\Psi(r_1, r_2, \cdots, r_N)$, if the particle are indistinguishable, the marginal density can be defined by

$$
n(r) = \int | \Phi(r_1, r_2, \cdots, r_N) |^2 dr_2 \cdots dr_N
$$

$n(r)$ is the probability density of the particle at position $r$.

The wave function must satisfy two specific conditions:

* **Normalization condition**:
    
    $$
    \int_{-\infty}^{\infty} |\Psi(r)|^2 dr = 1,
    $$

    since the interpretation of the wave function is that the square of it is a probability amplitude.

* **Anti-symmetry condition for fermions**:
    
    $$\hat P_{12}\Psi(r_1,r_2)= \Psi(r_2,r_1) = -\Psi(r_1,r_2).$$

    where $\hat P_{12}$ is the permutation operator, which swaps the position of the two particles.
