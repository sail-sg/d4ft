Exchange correlations functionals
=================================


.. py:function:: LDA(wave_fun, grid, weight, )

    .. math::
        E_{LDA} = - \frac{3}{4}\left( \frac{3}{\pi} \right)^{1/3}\int\rho(\mathbf{r})^{4/3}\ \mathrm{d}\mathbf{r}


    :param Callable wave_fun: a batched N-body wave function