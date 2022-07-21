"""WARNING: this module use pyscf.gto.module for initializing molecule object.

Example:
    # default ångström unit. need to change to Bohr unit.
    h2o_config = 'O 0 0 0; H 0 1 0; H 0 0 1'

    >>> from pyscf import gto
    >>> mol = gto.M(atom = h2o_config, basis = '3-21g')
    >>> mol.build()
    >>> mol._basis
    >>> {'H': [[0, [5.447178, 0.156285], [0.824547, 0.904691]],
              [0, [0.183192, 1.0]]],
    'O': [[0, [322.037, 0.0592394], [48.4308, 0.3515], [10.4206, 0.707658]],
          [0, [7.40294, -0.404453], [1.5762, 1.22156]],
          [0, [0.373684, 1.0]],
          [1, [7.40294, 0.244586], [1.5762, 0.853955]],
          [1, [0.373684, 1.0]]]}

    >>> mol.atom_coords()
    >>> array([[0.        , 0.        , 0.        ],
               [0.        , 1.88972612, 0.        ],
               [0.        , 0.        , 1.88972612]])


    >>> mol.atom_charges()
    >>> array([8, 1, 1], dtype=int32)

    # return the overlap (covariance) of MOs.
    >>> mol.intor('int1e_ovlp_sph').shape
    >>> (13, 13)   # 2+2+9 = 13

    >>> mol.tot_electrons()  # total number of electrons.
    >>> 10

    >>> mol.elements
    >>> ['O', 'H', 'H']

"""

import logging
import jax
from jax import random
import jax.numpy as jnp
from jax import vmap
from jdft.orbitals import Pople, PopleFast
from jdft.orbitals import MO_qr
from jdft.intor import Quadrature

from pyscf import gto
from pyscf.dft import gen_grid
from jdft.optimizer import sgd

logging.getLogger().setLevel(logging.INFO)


class molecule():
  """Class to represent a molecule."""

  def __init__(self, config, spin, basis='3-21g', level=1, eps=1e-10):
    """Initialize a molecule."""
    self.pyscf_mol = gto.M(atom=config, basis=basis, spin=spin)
    self.pyscf_mol.build()

    self.tot_electron = self.pyscf_mol.tot_electrons()
    self.elements = self.pyscf_mol.elements
    self.atom_coords = self.pyscf_mol.atom_coords()
    self.atom_charges = self.pyscf_mol.atom_charges()
    self.spin = spin  # number of non-paired electrons.
    self.nao = self.pyscf_mol.nao  # number of atomic orbitals
    self._basis = self.pyscf_mol._basis

    self.nocc = jnp.zeros([2, self.nao])  # number of occupied orbital.
    self.nocc = self.nocc.at[0, :int((self.tot_electron + self.spin) /
                                     2)].set(1)
    self.nocc = self.nocc.at[1, :int((self.tot_electron - self.spin) /
                                     2)].set(1)
    self.params = None
    self.tracer = []  # to store training curve.

    self.nuclei = {
      'loc': jnp.array(self.atom_coords),
      'charge': jnp.array(self.atom_charges)
    }
    self.level = level
    g = gen_grid.Grids(self.pyscf_mol)
    g.level = self.level
    g.build()
    self.grids = jnp.array(g.coords)
    self.weights = jnp.array(g.weights)
    self.eps = eps
    self.timer = []

    logging.info(
      'Initializing... %d grid points are sampled.', self.grids.shape[0]
    )
    logging.info('There are %d atomic orbitals in total.', self.nao)

    if self.nao >= 100:
      self.ao = PopleFast(self.pyscf_mol)
    else:
      self.ao = Pople(self.pyscf_mol)

    self.mo = MO_qr(
      self.nao, self.ao, Quadrature(None, self.grids, self.weights)
    )

  def _init_param(self, seed=123):
    key = random.PRNGKey(seed)
    return self.mo.init(key)

  def train(
    self,
    epoch,
    lr=1e-3,
    seed=123,
    converge_threshold=1e-3,
    batchsize=1000,
    save_fig=False,
    **args
  ):
   return sgd(
    self,
    epoch,
    lr,
    seed,
    converge_threshold,
    batchsize,
    save_fig,
    **args
   )

  def get_wave(self, occ_ao=True):
    """Calculate the wave function.

    occ_ao: if True, only return occupied orbitals.
    return: wave function value at grid_point.
    """

    def f(r):
      return self.mo(self.params, r) * self.nocc

    if occ_ao:
      alpha = vmap(f)(self.grids)[:, 0, :int(jnp.sum(self.nocc[0, :]))]
      beta = vmap(f)(self.grids)[:, 1, :int(jnp.sum(self.nocc[1, :]))]
      return jnp.concatenate((alpha, beta), axis=1)
    else:
      return vmap(f)(self.grids)

  def get_density(self, r):
    """Calculate the electron density at r.

    Returns: density function: [D, 3] -> [D] where D is the number of grids.
    """

    def f(r):
      return self.mo(self.params, r) * self.nocc

    wave = vmap(f)(r)  # (D, 2, N)
    alpha = wave[:, 0, :int(jnp.sum(self.nocc[0, :]))]
    beta = wave[:, 1, :int(jnp.sum(self.nocc[1, :]))]
    wave_all = jnp.concatenate((alpha, beta), axis=1)
    dens = jnp.sum(wave_all**2, axis=1)
    return dens


if __name__ == '__main__':
  h2o_geometry = """
  O 0.0000 0.0000 0.1173;
  H 0.0000 0.7572 -0.4692;
  H 0.0000 -0.7572 -0.4692;
  """

  h2o = molecule(h2o_geometry, spin=0)
  h2o.train(10)
