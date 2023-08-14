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
import string
from typing import Any

import shortuuid
from absl import app, flags, logging
from jax.config import config
from ml_collections.config_flags import config_flags

from d4ft.config import D4FTConfig
from d4ft.solver.drivers import (
  incore_cgto_direct_opt_dft,
  incore_cgto_pyscf_dft_benchmark,
  incore_cgto_scf_dft,
)
from d4ft.system.refdata import get_refdata_benchmark_set

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "run", "direct", ["direct", "scf", "pyscf", "reaction"],
  "which routine to run"
)
flags.DEFINE_string("reaction", "hf_h_hfhts", "the reaction to run")
flags.DEFINE_string("benchmark", "", "the refdata benchmark set to run")
flags.DEFINE_bool("use_f64", False, "whether to use float64")
flags.DEFINE_bool("pyscf", False, "whether to benchmark against pyscf results")
flags.DEFINE_bool("save", False, "whether to save results and trajectories")

config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_: Any) -> None:
  config.update("jax_enable_x64", FLAGS.use_f64)

  cfg: D4FTConfig = FLAGS.config
  print(cfg)

  if FLAGS.save:
    with cfg.unlocked():
      cfg.uuid = shortuuid.ShortUUID(
        alphabet=string.ascii_lowercase + string.digits
      ).random(8)

  if FLAGS.benchmark != "":
    assert FLAGS.save

    with cfg.unlocked():
      cfg.uuid = ",".join([FLAGS.benchmark, cfg.get_core_cfg_str(), cfg.uuid])

    cfg.save()

    systems, _, _ = get_refdata_benchmark_set(FLAGS.benchmark)
    for system in systems:
      with cfg.unlocked():
        cfg.mol_cfg.mol = "-".join([FLAGS.benchmark, system])

      try:
        if FLAGS.run == "direct":
          incore_cgto_direct_opt_dft(cfg, FLAGS.pyscf)
        else:
          raise NotImplementedError
      except Exception as e:
        logging.error(e)

    return

  if FLAGS.run == "direct":
    incore_cgto_direct_opt_dft(cfg, FLAGS.pyscf)

  elif FLAGS.run == "scf":
    incore_cgto_scf_dft(cfg)

  elif FLAGS.run == "pyscf":
    incore_cgto_pyscf_dft_benchmark(cfg)


if __name__ == "__main__":
  app.run(main)
