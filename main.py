"""Main wrapper for D4FT"""

import os

from absl import app, flags, logging
from jax.config import config

import d4ft
import d4ft.geometries

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 100000, "batch size")
flags.DEFINE_integer("epoch", 1000, "epoch")
flags.DEFINE_float("converge_threshold", 1e-8, "")
flags.DEFINE_float("lr", 1e-2, "learning rate for sgd")
flags.DEFINE_float("momentum", 5e-2, "momentum for scf")
flags.DEFINE_bool("pre_cal", False, "whether to pre-calculate the integrals")
flags.DEFINE_integer("seed", 137, "random seed")
flags.DEFINE_integer("spin", 0, "total spin")
flags.DEFINE_string("geometry", "he", "")
flags.DEFINE_string("opt", "adam", "optimizer")
flags.DEFINE_string("basis_set", "sto-3g", "which basis set to use")
flags.DEFINE_string("device", "0", "cuda visible device")
flags.DEFINE_bool("lr_decay", True, "whether to use a piecewise linear")
flags.DEFINE_string("xc", "lda", "exchange functional")
flags.DEFINE_integer("quad_level", 1, "number of the quadrature points")
flags.DEFINE_bool("use_f64", False, "whether to use float64")
flags.DEFINE_bool("debug_nans", False, "whether to enable nan debugging in jax")

flags.DEFINE_enum("algo", "sgd", ["sgd", "scf"], "which algorithm to use")


def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device

  logging.set_verbosity(logging.INFO)

  config.update("jax_enable_x64", FLAGS.use_f64)
  config.update("jax_debug_nans", FLAGS.debug_nans)

  geometry = getattr(d4ft.geometries, FLAGS.geometry + "_geometry")
  mol = d4ft.Molecule(
    geometry,
    spin=FLAGS.spin,
    level=FLAGS.quad_level,
    basis=FLAGS.basis_set,
    algo=FLAGS.algo,
    xc=FLAGS.xc
  )

  if FLAGS.algo == "sgd":
    d4ft.sgd(
      mol,
      epoch=FLAGS.epoch,
      lr=FLAGS.lr,
      batch_size=FLAGS.batch_size,
      converge_threshold=FLAGS.converge_threshold,
      pre_cal=FLAGS.pre_cal,
      optimizer=FLAGS.opt,
      seed=FLAGS.seed
    )

  elif FLAGS.algo == "scf":
    d4ft.scf(
      mol,
      epoch=FLAGS.epoch,
      seed=FLAGS.seed,
      momentum=FLAGS.momentum,
    )


if __name__ == '__main__':
  app.run(main)
