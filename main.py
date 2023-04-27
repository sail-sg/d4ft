"""Entrypoint for D4FT"""
import jax
from absl import app, flags
from ml_collections.config_flags import config_flags
from d4ft.system.mol import get_pyscf_mol
from d4ft.sgd_solver import sgd_solver

FLAGS = flags.FLAGS

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "dft", ["dft"], "which routine to run")
config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_):
  cfg = FLAGS.config  # type: ConfigDict
  print(cfg)

  if FLAGS.run == "dft":
    mol = get_pyscf_mol(cfg.mol_cfg.mol_name, cfg.mol_cfg.basis)
    key = jax.random.PRNGKey(cfg.optim_cfg.rng_seed)
    sgd_solver(cfg.dft_cfg, cfg.optim_cfg, mol, key)


if __name__ == "__main__":
  app.run(main)
