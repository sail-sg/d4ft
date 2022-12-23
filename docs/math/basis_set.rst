Basic sets
========================

Pople's split valence type
--------------------------

The single electron wave function $\phi_i$'s are usually approximated by a series of set functions:

.. math::
    \psi_i(\boldsymbol{r}) = \sum_\alpha C_{\alpha, i} g_\alpha(\boldsymbol{r})


where :math:`C_{\alpha, i}` is the coefficient of the :math:`\alpha`-th basis function for the :math:`i`-th molecular orbital. One of the most commonly-used basis function is the Gaussian-type orbitals (GTO). A Gaussian basis centered at :math:`\boldsymbol{r}'= (x', y', z')^\top` of order :math:`l, m, n` is defined by,

.. math::
    g_{l, m, n}(\boldsymbol{r}; \zeta, \boldsymbol{r}') = c(x-\boldsymbol{r}'_x)^l(y-\boldsymbol{r}'_y)^m(z-\boldsymbol{r}'_z)^n e^{-\zeta\Vert \boldsymbol{r}-\boldsymbol{r}'\Vert^2}

with normalization constant:

.. math::
    c = \left(  \dfrac{2\zeta}{\pi} \right)^{\frac34} \left( \dfrac{(8\zeta)^{l+m+n} l!m!n! }{(2l)!(2m)!(2n)!}\right)^{\frac12}.


