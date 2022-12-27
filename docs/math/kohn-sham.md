# Kohn-Sham method

The idea of the density functional theory, abbreviated in DFT, is to replace the minimization problem in the 
last section, defined in terms of the unknown wavefunction $\Psi$, with a minimization problem in terms of 
the unknown density $n$. 

The Kohn-Sham method is a method to solve the DFT problem. The Kohn-Sham method uses the independent particle approximation to the kinetic energy oprator, which is the same as the Hartree-Fock method. However, the Kohn-Sham method introduces an additional term, called exchange-correlation functional to remedy the error of the approximation.


For the sake of simplicity, we temporarily ignore the spin degree of freedom.

The minimization problem in the Kohn-Sham method is defined as

$$
\begin{align}
&\min_{\Psi_i}  \frac{1}{2}\sum_{i=1}^{N} \int \left | \nabla \Psi_i \right |^2+ \int n V + \frac12 \int \int \frac{n(x)n(y)}{|x - y|} + E_{\text{xc}}(n)\\
&s.t. \langle \Psi_i | \Psi_j \rangle = \delta_{ij}\\ 
\end{align}
$$


The minimizer of the Kohn-Sham method satisfies the following equation:

$$\left(  -\dfrac{1}{2}\nabla ^2 +  v_{\text{eff}} \right ) \psi_i =  \varepsilon_i \psi_i.$$

We can recover the Hamiltonian from the Kohn-Sham equation:

$$
\hat{H} = -\dfrac{1}{2}\nabla ^2 +  v_{\text{eff}}.
$$


$v^\sigma_{\text{eff}}$ is the effective potential, which is
defined by

$$v_{\text{eff}}(\boldsymbol{r})  = V_{\text{ext}}(\boldsymbol{r}) + \int \dfrac{n(\boldsymbol{r}')}{\vert \boldsymbol{r} - \boldsymbol{r}'\vert}  + v_{\text{xc}}(\boldsymbol{r}),$$

and

$$n(\boldsymbol{r}) = \sum_{i=1}^{N} k_{i} \big\vert{\psi_i(\boldsymbol{r} )}\big\vert^2,$$

in which $N$ is the total number of electrons, and
$k_i \in \{0, 1\}$ is the orbital occupation number satisfying

$$\sum_{i} k_i = N.$$
