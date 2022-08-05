import jax
import jax.numpy as jnp

# from jdft.orbitals import Pople, PopleFast
# from jdft.orbitals import MO_qr

from ao import Pople, PopleFast, Gaussian
from mo import MO_qr, MO_vpf
from vpf import VolumePreservingFlow
from absl import logging

from pyscf import gto
from pyscf.dft import gen_grid


class molecule():
  """Class to represent a molecule."""

  def __init__(
      self, config, spin, basis='3-21g', level=1, eps=1e-10, mode=None, **args
  ):
    """Initialize a molecule."""
    if basis == 'gaussian':
      self.pyscf_mol = gto.M(atom=config, basis='sto-3g', spin=spin)
    else:
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

    if basis == 'gaussian':
      self.ao = Gaussian(self.pyscf_mol)

    if mode == 'vpf':
      self.mo = MO_vpf(
          self.nao, self.ao, VolumePreservingFlow(layers=args['layers'])
      )
    else:
      self.mo = MO_qr(self.nao, self.ao)

  def _init_param(self, seed=123):
    key = jax.random.PRNGKey(seed)
    return self.mo.init(key)
