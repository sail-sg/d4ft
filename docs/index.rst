D4FT: A differantiable Jax implementation for density functional theory
=======================================================================


This project is a `JAX <https://github.com/google/jax/>`_-based package for differantiable density functional theory and quantum chemistry.



It has many great features for quantum chemistry calculation including:

* **deep learning**: our method is deep learning native; it is implemented in a widely-used deep learning framework JAX;

* **differentiable**: all the functions are differentiable, and thus the energy can be optimized via purely gradient-based methods;

* **direct optimization**: due to the differentiability, our method does not require self-consistent field iterations, but the convergence result is also self-consistent.


.. warning::

   This project is **under** development.  


.. .. important::
..    The guidance for development:

..    - **Involving more functional programming.**  We try to use pure functions and reduce side affects as much as possible. The only class in :guilabel:`D4FT` is ``jdft.molecule``. Functional programming is also one the most important characteristics of :guilabel:`Jax`
..    - **Align** :guilabel:`D4FT` **with** :guilabel:`PySCF`. The functions, attributes and methods are optimized to align with those in :guilabel:`PySCF`.

.. _installation:

Installation
------------

To use :guilabel:`D4FT`, first install it using pip:

.. code-block:: console

   $ pip install d4ft



Getting Started
###############

Currently we support the following calculation methods:

* vanilla self-consistent field method.
* stochastic self-consistent field method.
* stochasic gradient-based optimizers supported by `Optax <https://optax.readthedocs.io/en/latest/>`_.






Examples
########




Theory
########
We include a detailed mathematical derivation of density functional theory from scratch, which is useful for those who do not have related background. 


Support 
-------


The Team
########

This project is developed by `SEA AI LAB (SAIL) <https://sail.sea.com/>`_. We are also grateful to researchers from `NUS I-FIM <https://ifim.nus.edu.sg/>`_ for contributing ideas and theoretical support.  

.. image:: images/sail_logo.png
   :width: 300
   :alt: Alternative text

.. image:: images/ifim_logo.png
   :width: 300
   :alt: Alternative text


Citation
########

.. code-block:: console

 @article{li2022d4ft,
   title={D4FT}: A Deep Learning Approach to Kohn-Sham Density Functional Theory},
   author={Tianbo Li and Min Lin and Zheyuan Hu and Kunhao Zheng and Giovanni Vignale and Kenji Kawaguchi and A.H. Castro Neto and Kostya S. Novoselov and Shuicheng Yan},
   booktitle={},
   year={2022},
 }



License
-------



Contents
--------

.. toctree::
   :maxdepth: 2

   benchmark
   math/index
   d4ft/index
   examples/index


