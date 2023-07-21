# Copyright 2023 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import jax_xc
from absl import logging
from absl.testing import absltest, parameterized

from d4ft.integral.quadrature.utils import wave2density
from d4ft.config import get_config
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.quadrature.grids import DifferentiableGrids
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.utils import compose
from d4ft.xc import get_xc_intor
from d4ft.types import Fock, MoCoeff, QuadGridsNWeights


class XCTest(parameterized.TestCase):

  @parameterized.parameters(
    ("lda_x",),
    ("gga_x_pbe",),
  )
  def test_xc_grad(self, xc_name: str) -> None:
    xc_functional = getattr(jax_xc, xc_name)

    cfg = get_config()
    key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
    cfg.mol_cfg.mol = "h2"
    cfg.mol_cfg.basis = "sto-3g"

    # build system
    pyscf_mol = get_pyscf_mol(
      cfg.mol_cfg.mol, cfg.mol_cfg.basis, cfg.mol_cfg.spin, cfg.mol_cfg.charge,
      cfg.mol_cfg.geometry_source
    )
    mol = Mol.from_pyscf_mol(pyscf_mol)
    cgto = CGTO.from_mol(mol)

    # build quadrature grids
    dg = DifferentiableGrids(pyscf_mol)
    dg.level = cfg.direct_min_cfg.quad_level
    # TODO: test geometry optimization
    grids_and_weights = dg.build(pyscf_mol.atom_coords())

    # get overlap
    ovlp = pyscf_mol.intor('int1e_ovlp_sph')

    # function maps mo coefficients to xc energy
    mo_coeff_fn = partial(
      cgto.get_mo_coeff,
      rks=cfg.direct_min_cfg.rks,
      ortho_fn=qr_factor,
      ovlp_sqrt_inv=sqrt_inv(ovlp),
    )
    xc_fn = get_xc_intor(
      grids_and_weights,
      cgto,
      xc_functional,
      polarized=not cfg.direct_min_cfg.rks
    )

    mo_xc_fn = hk.without_apply_rng(hk.transform(compose(xc_fn, mo_coeff_fn)))
    params = mo_xc_fn.init(key)

    e_xc = mo_xc_fn.apply(params)
    logging.info(e_xc)
    self.assertFalse(jnp.isnan(e_xc))

    # test point
    r1 = grids_and_weights[0][0]

    def density_fn(mo_coeff: MoCoeff):
      mo_fn = lambda r: mo_coeff @ cgto.eval(r)
      density = wave2density(mo_fn, True)
      return jnp.sum(density(r1))

    mo_density_fn = hk.without_apply_rng(
      hk.transform(compose(density_fn, mo_coeff_fn))
    )
    n_r = mo_density_fn.apply(params)
    logging.info(n_r)

    # test gradient
    n_grads = jax.grad(mo_density_fn.apply)(params)
    logging.info(n_grads)
    self.assertFalse(jnp.isnan(n_grads["~"]["mo_params"]).any())

    xc_grads = jax.grad(mo_xc_fn.apply)(params)
    logging.info(xc_grads)
    self.assertFalse(jnp.isnan(xc_grads["~"]["mo_params"]).any())


if __name__ == "__main__":
  absltest.main()
