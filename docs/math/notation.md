The Bra-Ket Notation & Principles
====================

Here, we introduce the braket notation which represents the quantum state of the physical system. We limit to the case of a single particle in 3D space, and the notation can be extended to the case of multiple particles.

In quantum mechanics, a state of a physical system is described by a **wavefunction** $\displaystyle \boldsymbol{\Psi}: \mathbb{R}^{3} \to \mathbb{C}$, which is a complex-valued function of the position $\displaystyle \boldsymbol r$ of the system. The wavefunction is a **vector** in a Hilbert space $\displaystyle \mathcal{H}$, and the Hilbert space is a **vector space** over the complex numbers $\displaystyle \mathbb{C}$.

For a wavefunction $\displaystyle \boldsymbol{\Psi}$, we note it as $\displaystyle | \boldsymbol{\Psi} \rangle$ to represent it as a vector in $\displaystyle \mathcal{H}$. Dirac called these elements *kets*.

A *bra* is of the form $\displaystyle \langle \boldsymbol{\Phi}|$. It acts on the left-hand side of the inner-product in $\displaystyle \mathcal{H}$. Mathematically it denotes a linear form $\displaystyle \mathcal{H} \to \mathbb {C}$, i.e. **a linear map** that maps each wavefunction in $\displaystyle \mathcal{H}$ to a number in the complex plane $\displaystyle \mathbb {C}$. Letting the *bra* $\displaystyle \langle \boldsymbol{\Phi}|$ act on a *ket* $\displaystyle |\boldsymbol{\Psi}\rangle$ is written as $\displaystyle \langle \boldsymbol{\Phi}|\boldsymbol{\Psi}\rangle \in \mathbb {C}$. 

bra–ket notation, or Dirac notation, is used ubiquitously to denote quantum states. It is a shorthand notation for the inner-product of the wavefunction. Writing
$\displaystyle \langle \boldsymbol{\Psi}|\boldsymbol{\Psi}\rangle$ is equivalent to writing the square of the norm of the vector representing the wavefunction $\displaystyle \boldsymbol{\Psi}$, *i.e.* $\displaystyle \langle \boldsymbol{\Psi}|\boldsymbol{\Psi}\rangle = \int \boldsymbol{\Psi}^*( \boldsymbol r)\boldsymbol{\Psi} ( \boldsymbol r) d \boldsymbol r$.

#### Bras and kets as row and column vectors

Ideally, we would like to work on the Hilbert space $\displaystyle \mathcal{H}$ directly, but its infinite-dimensional nature
makes it difficult for machine computation. 

Instead, we would like to work on a finite-dimensional subspace, which is spanned by a set of basis function $\displaystyle \{ |g_i\rangle \}$, where $\displaystyle |g_i\rangle$ 
is a basis vector in $\displaystyle \mathcal{H}$. When the basis functions are chosen, we can represent the wavefunction $\displaystyle \boldsymbol{\Psi}$ as a linear combination of the basis functions.

For example, if we have a system with $N$ basis functions $g_i$, the wavefunction $\displaystyle \boldsymbol{\Psi}$ can be written as:

$$
\begin{equation}
  | \boldsymbol{\Psi} \rangle = \sum_{i=1}^N a_i | g_i \rangle.
\end{equation}
$$

where $|\mathbb{\Psi}\rangle$ is called a "state vector", and $a_i$ are the coefficients, and $|g_i\rangle$ are the orthonormal **basis** vectors. Under this basis, the state vector $|\mathbb{\Psi}\rangle$ can be written as a column vector:

$$
\begin{equation}
    | \boldsymbol{\Psi} \rangle=\begin{pmatrix}
    a_1 \\
    a_2\\
    \vdots \\
    a_N
    \\
    \end{pmatrix}.
\end{equation}
$$

and the bra notation $\langle \boldsymbol{\Psi} |$ denotes the conjugate transpose of $| \boldsymbol{\Psi} \rangle$, which is

$$
    \langle \boldsymbol{\Psi} | = (a_1^*, a_2^*, \dots, a_N^*).
$$

The inner product of two state vectors can be therefore written as

$$
\begin{align}
\langle \boldsymbol{\Psi}_a | \boldsymbol{\Psi}_b \rangle &= \int_{\mathbb{R}^3} \boldsymbol{\Psi}^*_a(\boldsymbol r)\boldsymbol{\Psi}_b( \boldsymbol x)d\boldsymbol{r}\\
&= \int_{\mathbb{R}^3} \sum_i \sum_j a_i^* b_j g_i(\boldsymbol{r})  g_j(\boldsymbol{r})  d\boldsymbol{r}   \\
&= \sum_i a_i^*b_i. 
\end{align}
$$

#### Expectation of a physical quantity

In quantum mechanics, each physical quantity is associated with a self-adjoint(Hermitian) 
**operator** $\hat A$ (physicians call it *observable*), which is a linear map from $\mathcal{H}
$ to $\mathcal{H}$, *i.e.* $\hat A: \mathcal{H} \to \mathcal{H}$. 

For example, we associate the position operator $\hat x$ with the position of a particle, and the momentum operator $\hat p$ with the momentum of a particle, the Hamiltonian operator $\hat H$ with the energy of a particle, etc.

To obtain a certain physial quantity out of a state $| \boldsymbol{\Psi} \rangle$, we need to calculate the expectation of the operator $\hat A$ with respect to $| \boldsymbol{\Psi} \rangle$, which is defined as the expectation value of $\hat A$ in the state $| \boldsymbol{\Psi} \rangle$.

For example, to obtain the energy of a particle in a state $| \boldsymbol{\Psi} \rangle$, we need to calculate the expectation value of the Hamiltonian operator $\hat H$ with respect to $| \boldsymbol{\Psi} \rangle$.

Suppose we have an operator $\hat A$, the **expectation** of $\hat A$ is defined as,

$$
\begin{align}
\langle \hat A \rangle_\Psi := \langle \boldsymbol{\Psi} | \hat A | \boldsymbol{\Psi} \rangle 
=  \int_{\mathbb{R}^3} \boldsymbol{\Psi}^*(\boldsymbol r) \left( \hat A\boldsymbol{\Psi}( \boldsymbol r) \right) d\boldsymbol{r}.
\end{align}
$$

If $| \boldsymbol{\Psi} \rangle$ happens to be an eigenstate of $\hat A$, then the expectation value of $\hat A$ is simply the eigenvalue of $\hat A$ in that state.
