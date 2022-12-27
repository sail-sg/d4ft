# Exchange-correlation functional

The exchange-correlation functional is a functional of the density $n$. Its goal is to describe the exchange-correlation between electrons due to the fact that the kinetic energy operator is under the independent particle approximation.

The analytical form of the exchange-correlation functional is unknown. Over the years, many approximations have been proposed. The most popular ones are the local density approximation (LDA), the generalized gradient approximation (GGA), the meta-GGA, and the hybrid functional.

We follows the notation in libxc, which is a library for exchange-correlation functionals. The exchange-correlation functional is denoted as $E_{\text{xc}}(n)$.

For LDAs, the general form of the exchange-correlation functional is given by

$$
E_{\text{xc}}(n) = \int \varepsilon_{\text{xc}}(n(\boldsymbol{r}))n(\boldsymbol{r}) \, d \boldsymbol{r}.
$$

The $\varepsilon_{\text{xc}}$ is a function that transforms the density $n(\boldsymbol{r})$ into the exchange-correlation energy density $\varepsilon_{\text{xc}}(n(\boldsymbol{r}))$.

For GGAs, the general form of the exchange-correlation functional is given by

$$
E_{\text{xc}}(n) = \int \varepsilon_{\text{xc}}(n(\boldsymbol{r}), \nabla n(\boldsymbol{r}))n(\boldsymbol{r}) \, d \boldsymbol{r}.
$$

The $\varepsilon_{\text{xc}}$ depends on the density $n(\boldsymbol{r})$ and the gradient of the density $\nabla n(\boldsymbol{r})$.

For meta-GGAs, the general form of the exchange-correlation functional is given by

$$
E_{\text{xc}}(n) = \int \varepsilon_{\text{xc}}(n(\boldsymbol{r}), \nabla n(\boldsymbol{r}), \nabla^2 n(\boldsymbol{r}), \tau(r))n(\boldsymbol{r}) \, d \boldsymbol{r}.
$$

Here the $\tau(r)$ is the kinetic energy density, which is defined by

$$
\tau(r) = \frac{1}{2} \sum_i^{N} |\nabla \psi_i(r)|^2.
$$
