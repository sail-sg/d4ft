"""Entrypoint for D4FT."""
from functools import partial

import jax
from absl import app, flags
from ml_collections.config_flags import config_flags

from d4ft.hamiltonian.cgto_intors import get_cgto_intor
from d4ft.hamiltonian.dft_cgto import dft_cgto
from d4ft.hamiltonian.ortho import qr_factor
from d4ft.integral.gto.cgto import CGTO
from d4ft.integral.obara_saika.driver import incore_int_sym
from d4ft.sgd_solver import sgd_solver
from d4ft.system.mol import Mol
from d4ft.xc import get_xc_intor

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "dft", ["dft"], "which routine to run")
config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict
  print(cfg)

  if FLAGS.run == "dft":
    key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
    # TODO: consider separate grids
    mol = Mol.from_mol_name(
      cfg.mol_cfg.mol_name, cfg.mol_cfg.basis, cfg.dft_cfg.quad_level
    )
    cgto = CGTO.from_mol(mol, use_hk=False)
    incore_energy_tensors = incore_int_sym(cgto)
    cgto_intor = get_cgto_intor(
      cgto, intor="obsa", incore_energy_tensors=incore_energy_tensors
    )
    xc_fn = get_xc_intor(mol, cfg.dft_cfg.xc_type)
    mo_coeff_fn = partial(
      mol.get_mo_coeff, rks=cfg.dft_cfg.rks, ortho_fn=qr_factor
    )
    H_fac = partial(dft_cgto, cgto, cgto_intor, xc_fn, mo_coeff_fn)
    sgd_solver(cfg.dft_cfg, cfg.optim_cfg, H_fac, key)


if __name__ == "__main__":
  app.run(main)
