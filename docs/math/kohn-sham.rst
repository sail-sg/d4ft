Kohn-Sham method
================


The Kohn-Sham equation:

.. math::
  \left[  -\dfrac{\hbar}{2m}\nabla ^2 +  v^\sigma_{eff}(\boldsymbol{r})  \right] \psi_i^\sigma (\boldsymbol{r}) =  \epsilon \psi^\sigma_i(\boldsymbol{r} )

where :math:`\sigma \in \{\uparrow, \downarrow\}` is the spin direction, :math:`v^\sigma_{eff}(\boldsymbol{r})`  is the effective potential, which is defined by

.. math::
  v^\sigma_{eff}(\boldsymbol{r})  = V^\sigma_{ext}(\boldsymbol{r}) + \int d^3 \boldsymbol{r}' \dfrac{n(\boldsymbol{r}')}{\vert \boldsymbol{r} - \boldsymbol{r}'\vert}  + v^\sigma_{xc}(\boldsymbol{r}),

and

.. math::
  n(\boldsymbol{r}) = \sum_{\sigma}\sum_{i=1}^{N_e} k^\sigma_{i} \big\vert{\psi^\sigma_i(\boldsymbol{r} )}\big\vert^2

in which :math:`N_e` is the total number of electrons, and :math:`k_i^\sigma \in [0, 1]` is the orbital occupation number satisfying

.. math::
    \sum_{\sigma}\sum_{i} k_i^\sigma = N_e

