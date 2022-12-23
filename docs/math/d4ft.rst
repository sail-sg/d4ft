D4FT method
================


Background
----------



The objective for Kohn-sham equation with gradient descent
----------------------------------------------------------

The Kohn-Sham Equation eventually converge until

.. math::
  \left[  -\dfrac{1}{2}\nabla ^2 + V^\sigma_{ext}(\boldsymbol{r}) + \int d^3 \boldsymbol{r}' \dfrac{n^\sigma(\boldsymbol{r}')}{\vert \boldsymbol{r} - \boldsymbol{r}'\vert}  + v^\sigma_{xc}(\boldsymbol{r}) \right] \psi_i^\sigma (\boldsymbol{r}) =  \varepsilon_i \psi^\sigma_i(\boldsymbol{r} )

instead of fixing :math:`n(\boldsymbol{r}')`, we substitute :math:`n^\sigma(\boldsymbol{r}')=\sum \vert \psi^\sigma_i(\boldsymbol{r}') \vert^2` and we have

.. math::
    \epsilon_i = \min_{\psi^\sigma_i} \left<\psi^\sigma_i \left\vert -\dfrac{1}{2}\nabla ^2 + v^\sigma_{ext}(\boldsymbol{r}) + \int d^3 \boldsymbol{r}' \dfrac{\sum \vert \psi^\sigma_i(\boldsymbol{r}') \vert^2}{\vert \boldsymbol{r} - \boldsymbol{r}'\vert}  + v^\sigma_{xc}(\boldsymbol{r}) \right\vert \psi^\sigma_i \right>


