Molecule
========

.. py:class:: jdft.molecule

  ``jdft.molecule`` is the core object for the computation.


  .. py:function:: init(self, config, spin:int, basis:str, )

    :param List config:  the configuration of the molecule. spin the number of.
    :param int spin:  the number of unpaired electrons, the difference between the number of alpha and beta electrons.
    :param str basis: label for basis set. Implemented basis sets can be found 


  .. py:attribute:: config

    the configuration of molecule

  .. py:function:: train(self, lr:float, )

    :param float lr:

