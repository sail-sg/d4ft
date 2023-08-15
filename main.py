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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shortuuid
from absl import app, flags, logging
from jax.config import config
from ml_collections.config_flags import config_flags

from d4ft.config import D4FTConfig
from d4ft.constants import HARTREE_TO_KCAL_PER_MOL
from d4ft.solver.drivers import (
  incore_cgto_direct_opt_dft,
  incore_cgto_pyscf_dft_benchmark,
  incore_cgto_scf_dft,
)
from d4ft.system.refdata import get_refdata_benchmark_set

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "run", "direct", ["direct", "scf", "pyscf", "reaction", "viz"],
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
      if system == "bh76_h":  # TODO: fix the xc grad NaN issue
        continue

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

  elif FLAGS.run == "viz":
    p = Path(cfg.save_dir)
    runs = [f for f in p.iterdir() if f.is_dir()]
    direct_df = pd.DataFrame()
    pyscf_df = pd.DataFrame()
    for run in runs:
      direct_df_i = pd.read_csv(run / "direct_opt.csv").iloc[-1:]
      pyscf_df_i = pd.read_csv(run / "pyscf.csv")
      direct_df_i.index = [run.name]
      pyscf_df_i.index = [run.name]
      direct_df = pd.concat([direct_df, direct_df_i])
      pyscf_df = pd.concat([pyscf_df, pyscf_df_i])

    direct_df = direct_df.rename(columns={
      'Unnamed: 0': 'steps',
    })
    pyscf_df = pyscf_df.rename(columns={
      'Unnamed: 0': 'steps',
    })
    diff_df = (direct_df - pyscf_df) * HARTREE_TO_KCAL_PER_MOL
    diff_df['e_total'].dropna().sort_values().plot(kind='bar')
    plt.title("BH76 benchmark set energy difference (direct - pyscf)")
    plt.ylabel("dE (kcal/mol)")
    plt.plot(diff_df.index, [1] * len(diff_df.index), 'r--')
    plt.show()


if __name__ == "__main__":
  app.run(main)
