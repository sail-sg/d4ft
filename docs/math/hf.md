# Hartree-Fock Method

Hatree-Fock Method is the method that approximates the wavefunction of a multi-particle system by hypothesizing that all particles are *independent*.

## Slater Determinant

For a system with $n$ particles, the wavefunction is a function of $n$ coordinates, $\Psi(\boldsymbol{r}_1,\boldsymbol{r}_2,\cdots,\boldsymbol{r}_n)$. In the case where all particles are independent, the wavefunction can be written as a product of wavefunctions of each particle, $\Psi(\boldsymbol{r}_1,\boldsymbol{r}_2,\cdots,\boldsymbol{r}_n)=\prod_{i=1}^n\Psi_i(\boldsymbol{r}_i)$. 

Applying the antisymmetrizer to the above function, we recover the definition of a determinant. In the language of quantum chemistry, this determinant is called a *Slater Determinant*, up to a normalized constant $\sqrt{n!}$.

$$
\begin{align}
\mathcal{A}\Psi(\boldsymbol{r}_1,\boldsymbol{r}_2,\cdots,\boldsymbol{r}_n)&=\sum_{\pi\in\Pi}(-1)^{\mathrm{inv}(\pi)}\prod_{i=1}^n\Psi_i(\boldsymbol{r}_{\pi(i)}) \\
&= \det \left(\Psi_i(\boldsymbol{r}_j)\right) \\
&= \begin{vmatrix}
\Psi_1(\boldsymbol{r}_1) & \Psi_1(\boldsymbol{r}_2) & \cdots & \Psi_1(\boldsymbol{r}_n) \\
\Psi_2(\boldsymbol{r}_1) & \Psi_2(\boldsymbol{r}_2) & \cdots & \Psi_2(\boldsymbol{r}_n) \\
\vdots & \vdots & \ddots & \vdots \\
\Psi_n(\boldsymbol{r}_1) & \Psi_n(\boldsymbol{r}_2) & \cdots & \Psi_n(\boldsymbol{r}_n) \\
\end{vmatrix}
\end{align}
$$

By assuming that all particles are independent, Hartree-Fock method limits the search space of the wavefunction to the form of a Slater Determinant. We also add a constraint that the single-particle wavefunctions are orthonormal, *i.e.* $\langle\Psi_i|\Psi_j\rangle=\delta_{ij}$.

## Fock Operator

Under the assumption that all particles are independent, the minimization problem can be re-written as follows:

$$
\begin{align}
\min_{\Psi_i}&\langle\det \left(\Psi_i(\boldsymbol{r}_j)\right)|\hat{H}|\det \left(\Psi_i(\boldsymbol{r}_j)\right)\rangle\\
s.t.\;& \langle\Psi_i|\Psi_j\rangle=\delta_{ij}
\end{align}
$$

The minimizer of the above problem satisfies the associated euler-Lagrange equations, namely the following system of $n$ coupled PDEs:

$$
\begin{align}
& - \frac12 \nabla^2\Psi_i + V \Psi_i + \left ( \sum_{j=1}^n |\Psi_j|^2 \star \frac{1}{|x|} \right ) \Psi_i - \sum_{j=1}^n \left ( \Psi_i\Psi_j \star \frac1{|x|} \right ) \Psi_j = \lambda_{i} \Psi_i \\
&\langle\Psi_i|\Psi_j\rangle = \delta_{ij}
\end{align}
$$

By rewriting the equation into the form of

$$
\hat F\Psi_i = \lambda_i \Psi_i,
$$

we define the Fock operator $\hat F$ as the sum of the kinetic energy operator, the nuclear attraction operator, and the electron-electron repulsion operator.
