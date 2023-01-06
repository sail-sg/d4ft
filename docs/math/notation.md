The Bra-Ket Notation & Principles
====================

In quantum mechanics, braket notation is a way of representing the wave function and the operators that act on it.
It is a powerful tool that allows us to mathematically describe the behavior of particles in a quantum system.

Here, we limit to the case of a single particle in 3D space, and the notation can be extended to the case of multiple particles.

In quantum mechanics, a state of a physical system is described by a **wavefunction** $\displaystyle \boldsymbol{\Psi}: \mathbb{R}^{3} \to \mathbb{C}$, a complex-valued function of the position $\displaystyle \boldsymbol r$ of the system. 

Mathematically, we say that the wavefunction is a **vector** in a Hilbert space $\displaystyle \mathcal{H}$. And from now on, we may use the term *wavefunction* and *state vector* interchangeably.

The Hilbert space is a **vector space** that consists of a set of wave functions, known as state vectors, that describe the possible states of the system. 

By using the language of Hilbert space, we can work with the functions in a similar way to how we work with vectors. For AI folks, dealing with vectors is a familiar concept (like in CV, NLP, etc.), and we can think of the Hilbert space as a vector space that is infinite-dimensional.

The braket notation is composed of two part: the *ket* and the *bra*. The ket is used to represent the wavefunction(vector), and the bra is used to represent the conjugate of the wavefunction.

For a wavefunction $\displaystyle \boldsymbol{\Psi}$, we note it as $\displaystyle | \boldsymbol{\Psi} \rangle$ to represent it as a vector in $\displaystyle \mathcal{H}$. Dirac called these elements *kets*.

A *bra* is of the form $\displaystyle \langle \boldsymbol{\Phi}|$. It acts on the left-hand side of the inner-product in $\displaystyle \mathcal{H}$. Mathematically it denotes a linear form $\displaystyle \mathcal{H} \to \mathbb {C}$, i.e. **a linear map** that maps each wavefunction in $\displaystyle \mathcal{H}$ to a scalar. Letting the *bra* $\displaystyle \langle \boldsymbol{\Phi}|$ act on a *ket* $\displaystyle |\boldsymbol{\Psi}\rangle$ is written as $\displaystyle \langle \boldsymbol{\Phi}|\boldsymbol{\Psi}\rangle \in \mathbb {C}$. 

To perform calculations using braket notation, we use the inner product. The innfer product is a mathematical operation that takes two wave functions and produces a scalar value. It is used to find the overlap between two wave functions or to calculate the expectation value of an operator.

bra–ket notation, or Dirac notation, is used ubiquitously to denote quantum states. It is a shorthand notation for the inner-product of the wavefunction. Writing
$\displaystyle \langle \boldsymbol{\Phi}|\boldsymbol{\Psi}\rangle$ just means you are doing $\displaystyle \langle \boldsymbol{\Phi}|\boldsymbol{\Psi}\rangle = \int \boldsymbol{\Phi}^*( \boldsymbol r)\boldsymbol{\Psi} ( \boldsymbol r) d \boldsymbol r$.


#### Bras and kets as row and column vectors

Ideally, we would like to work on the Hilbert space $\displaystyle \mathcal{H}$ directly, but its infinite-dimensional nature
makes it difficult for machine computation. 

Instead, we would like to work on a finite-dimensional subspace, which is spanned by a set of basis function $\displaystyle \{ |g_i\rangle \}$, where $\displaystyle |g_i\rangle$ 
is a basis vector in $\displaystyle \mathcal{H}$. In this subspace, we can represent the wavefunction $\displaystyle \boldsymbol{\Psi}$ as a linear combination of the basis functions.

For example, if we have a system with $N$ basis functions $g_i$, the wavefunction $\displaystyle \boldsymbol{\Psi}$ can be written as:

$$
\begin{equation}
  | \boldsymbol{\Psi} \rangle = \sum_{i=1}^N a_i | g_i \rangle.
\end{equation}
$$

Here, $a_i$ are the coefficients. Ideally, we want the **basis** vectors $|g_i\rangle$ to be orthonormal. Under this basis, the state vector $|\mathbb{\Psi}\rangle$ can be written as a column vector:

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

This is quite similar to the inner product of two vectors in $\mathbb{R}^N$. The simplicity of the result is due to the fact that the basis functions are orthonormal.

#### Operator and Measurement

In quantum mechanics, operators are mathematical functions that act on the wave function of a quantum system to produce a new wave function.

When an operator acts on a wavefunction, it is written as a product of the operator and the wavefunction. For example, $\hat A |\boldsymbol{\Psi}\rangle$ represents the action of the operator $\hat A$ on the wave function $|\boldsymbol{\Psi}\rangle$.

Similar to linear algebra, you can think of the operator as a matrix, and the wave function as a column vector.

Each physical quantity is associated with a self-adjoint(Hermitian) 
**operator** $\hat A$ (physicians call it *observable*), which is a linear map from $\mathcal{H}
$ to $\mathcal{H}$, *i.e.* $\hat A: \mathcal{H} \to \mathcal{H}$. The property of self-adjointness says that the eigenvalues of $\hat A$ are real numbers. 

Similar to the eigenvalues and eigenvectors of a matrix, the eigenvalues of an operator are called **eigenvalues** and the corresponding eigenvectors are called **eigenstates**. The **eigenstates** is a set of speicial wavefunctions that are orthogonal to each other.

For example, we associate the position operator $\hat x$ with the position of a particle, and the momentum operator $\hat p$ with the momentum of a particle, the Hamiltonian operator $\hat H$ with the energy of a particle, etc. 

The fundamental principle of quantum mechanics states that after a measurement of the physical quantity (e.g. energy), the system will be in one of the eigenstates of the operator $\hat A$(e.g. $\hat H$) with the corresponding eigenvalue (e.g. measured energy). 
The probability of the system being in the eigenstate $| \boldsymbol{\Psi} \rangle$ is proportional to the square of the amplitude of the state vector in that component.

Let's see an example below:

Suppose in our case that the Hamiltonian operator $\hat H$ has only two eigenvalues $E_1$ and $E_2$, and the corresponding eigenstates are $| \boldsymbol{\Psi}_1 \rangle$ and $| \boldsymbol{\Psi}_2 \rangle$. It means that for any state, after a measurement of the energy, the system will be in one of the two states $| \boldsymbol{\Psi}_1 \rangle$ or $| \boldsymbol{\Psi}_2 \rangle$. The measured energy will be either $E_1$ or $E_2$ and no other value is possible.

For a particle in a state $a_1 | \boldsymbol{\Psi}_1 \rangle + a_2 | \boldsymbol{\Psi}_2 \rangle$, (we assume that it is already normalized, *i.e.* $a_1^2 + a_2^2 = 1$), the probability of the particle being in the state $| \boldsymbol{\Psi}_1 \rangle$ is $|a_1|^2$, and the probability of the particle being in the state $| \boldsymbol{\Psi}_2 \rangle$ is $|a_2|^2$.

In other words, after a measurement of the energy, the probability of we get the energy $E_1$ is $|a_1|^2$, and the probability of we get the energy $E_2$ is $|a_2|^2$.

However, if the particle is in one of the eigenstates, the probability of getting the corresponding eigenvalue is 1, and the probability of getting the other eigenvalue is 0. Only when the particle is in one of the eigenstates, we know *a priori* the value of the energy obtained from the measurement.

#### Expectation of a physical quantity

Due to the probabilistic nature of quantum mechanics, we do not talk about the value of a physical quantity, but the **expectation** of the physical quantity, given a state $| \boldsymbol{\Psi} \rangle$.

To obtain the expectation of the operator $\hat A$ with respect to $| \boldsymbol{\Psi} \rangle$, we need to calculate the inner product of $\langle \boldsymbol{\Psi} |$ and $\hat A | \boldsymbol{\Psi} \rangle$, that is, $\langle \boldsymbol{\Psi} | \hat A | \boldsymbol{\Psi} \rangle$.

Suppose we have an operator $\hat A$, the **expectation** of $\hat A$ is defined as,

$$
\begin{align}
\langle \hat A \rangle_\Psi := \langle \boldsymbol{\Psi} | \hat A | \boldsymbol{\Psi} \rangle 
=  \int_{\mathbb{R}^3} \boldsymbol{\Psi}^*(\boldsymbol r) \left( \hat A\boldsymbol{\Psi}( \boldsymbol r) \right) d\boldsymbol{r}.
\end{align}
$$

If $| \boldsymbol{\Psi} \rangle$ happens to be an eigenstate of $\hat A$, then the expectation value of $\hat A$ is simply the eigenvalue of $\hat A$ in that state.
