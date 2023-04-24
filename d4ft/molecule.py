# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from absl import logging
from pyscf import gto
from pyscf.dft import gen_grid

from d4ft.ao import Gaussian, PopleFast
from d4ft.mo import MO, MO_qr


class Molecule:
  """Class to represent a molecule."""

  def __init__(
    self,
    geometry: str,
    spin: int = 0,
    basis: str = '3-21g',
    restricted_mo: bool = True,
    xc: str = 'lda',
    level: int = 1,
    seed: int = 123,
    eps: float = 1e-10,
    algo: str = None,
    **kwargs
  ):
    """Initialize a molecule."""
    if basis == 'gaussian':
      self.pyscf_mol = gto.M(atom=geometry, basis='sto-3g', spin=spin)
    else:
      self.pyscf_mol = gto.M(atom=geometry, basis=basis, spin=spin)

    self.pyscf_mol.build()

    self.tot_electron = self.pyscf_mol.tot_electrons()
    self.elements = self.pyscf_mol.elements
    self.atom_coords = self.pyscf_mol.atom_coords()
    self.atom_charges = self.pyscf_mol.atom_charges()
    self.atom_symbol = [
      self.pyscf_mol.atom_symbol(i) for i in range(self.pyscf_mol.natm)
    ]
    self.spin = spin  # number of non-paired electrons.
    self.nao = self.pyscf_mol.nao  # number of atomic orbitals
    self._basis = self.pyscf_mol._basis
    self.basis = basis
    self.restricted_mo = restricted_mo
    self.xc = xc

    self.nocc = jnp.zeros([2, self.nao])  # number of occupied orbital.
    self.nmo_up = (self.tot_electron + self.spin) // 2
    self.nmo_dn = (self.tot_electron - self.spin) // 2
    self.nocc = self.nocc.at[0, :self.nmo_up].set(1)
    self.nocc = self.nocc.at[1, :self.nmo_dn].set(1)

    self.nuclei = {
      'loc': jnp.array(self.atom_coords),
      'charge': jnp.array(self.atom_charges)
    }

    self.level = level
    # self.grids, self.weights = _gen_grid(self.pyscf_mol, level)
    g = gen_grid.Grids(self.pyscf_mol)
    g.level = self.level
    g.build()
    self.grids = jnp.array(g.coords)
    self.weights = jnp.array(g.weights)

    self.eps = eps
    self.seed = seed
    self.timer = []

    logging.info(
      'Initializing... %d grid points are sampled.', self.grids.shape[0]
    )
    logging.info('There are %d atomic orbitals in total.', self.nao)

    # build ao
    self.ao = PopleFast(self.pyscf_mol)

    if basis == 'gaussian':
      self.ao = Gaussian(self.pyscf_mol)

    # build mo
    if algo == 'sgd':
      self.mo = MO_qr(self.nao, self.ao, self.restricted_mo)

    else:
      self.mo = MO(self.nao, self.ao, self.restricted_mo)

  def _init_param(self, seed=None):
    seed = seed if seed else self.seed
    key = jax.random.PRNGKey(seed)
    return self.mo.init(key)
