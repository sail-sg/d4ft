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
"""Entrypoint for D4FT."""
from functools import partial
from typing import Any, Callable, Tuple

import jax
from absl import app, flags, logging
from jax_xc import lda_x
from ml_collections.config_flags import config_flags

from d4ft.config import D4FTConfig
from d4ft.hamiltonian.cgto_intors import get_cgto_intor
from d4ft.hamiltonian.dft_cgto import dft_cgto
from d4ft.hamiltonian.ortho import qr_factor, sqrt_inv
from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import incore_int_sym
from d4ft.integral.quadrature.grids import grids_from_pyscf_mol
from d4ft.logger import RunLogger
from d4ft.solver.pyscf_wrapper import pyscf
from d4ft.solver.sgd import sgd
from d4ft.system.mol import Mol, get_pyscf_mol
from d4ft.types import Hamiltonian
from d4ft.utils import make_constant_fn
from d4ft.xc import get_xc_intor

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "dft", ["dft", "pyscf"], "which routine to run")
config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_: Any) -> None:
  cfg: D4FTConfig = FLAGS.config
  print(cfg)

  if FLAGS.run == "dft":
    key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
    pyscf_mol = get_pyscf_mol(cfg.mol_cfg.mol_name, cfg.mol_cfg.basis)
    # TODO: change this to use obsa
    ovlp = pyscf_mol.intor('int1e_ovlp_sph')
    mol = Mol.from_pyscf_mol(pyscf_mol)
    grids_and_weights = grids_from_pyscf_mol(
      pyscf_mol, cfg.dft_cfg.quad_level
    )  # TODO: replace this with differentiable grids
    cgto = CGTO.from_mol(mol)

    # TODO: intor.split() for pmap / batched
    s2 = obsa.angular_static_args(*[cgto.primitives.angular] * 2)
    s4 = obsa.angular_static_args(*[cgto.primitives.angular] * 4)
    incore_energy_tensors = incore_int_sym(cgto, s2, s4)

    def H_factory() -> Tuple[Callable, Hamiltonian]:
      # TODO: out-of-core + basis optimization
      # cgto_hk = cgto.to_hk()
      cgto_hk = cgto
      cgto_intor = get_cgto_intor(
        cgto_hk, intor="obsa", incore_energy_tensors=incore_energy_tensors
      )
      mo_coeff_fn = partial(
        cgto_hk.get_mo_coeff,
        rks=cfg.dft_cfg.rks,
        ortho_fn=qr_factor,
        ovlp_sqrt_inv=sqrt_inv(ovlp),
      )
      xc_fn = get_xc_intor(grids_and_weights, cgto_hk, lda_x)
      return dft_cgto(cgto_hk, cgto_intor, xc_fn, mo_coeff_fn)

    sgd(cfg.dft_cfg, cfg.optim_cfg, H_factory, key)

  elif FLAGS.run == "pyscf":

    # solve with pyscf
    pyscf_mol = get_pyscf_mol(cfg.mol_cfg.mol_name, cfg.mol_cfg.basis)
    mo_coeff = pyscf(
      pyscf_mol, cfg.dft_cfg.rks, cfg.dft_cfg.xc_type, cfg.dft_cfg.quad_level
    )

    mol = Mol.from_pyscf_mol(pyscf_mol)
    cgto = CGTO.from_mol(mol)

    s2 = obsa.angular_static_args(*[cgto.primitives.angular] * 2)
    s4 = obsa.angular_static_args(*[cgto.primitives.angular] * 4)

    incore_energy_tensors = incore_int_sym(cgto, s2, s4)
    cgto_intor = get_cgto_intor(
      cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
    )
    grids_and_weights = grids_from_pyscf_mol(pyscf_mol, cfg.dft_cfg.quad_level)
    xc_fn = get_xc_intor(grids_and_weights, cgto, lda_x)

    # add spin and apply occupation mask
    mo_coeff *= mol.nocc[:, :, None]
    mo_coeff = mo_coeff.reshape(-1, mo_coeff.shape[-1])

    # eval with d4ft
    _, H = dft_cgto(
      cgto, cgto_intor, xc_fn, mo_coeff_fn=make_constant_fn(mo_coeff)
    )

    _, (energies, _) = H.energy_fn(mo_coeff)

    logger = RunLogger()
    logger.log_step(energies, 0)
    logger.get_segment_summary()
    logging.info(f"1e energy:{energies.e_kin + energies.e_ext}")


if __name__ == "__main__":
  app.run(main)
