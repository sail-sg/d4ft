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
import pickle
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
  incore_cgto_direct_opt,
  incore_cgto_pyscf_benchmark,
  incore_cgto_scf,
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
flags.DEFINE_bool(
  "basis_optim", False, "whether to optimize contraction coeff of the GTO basis"
)
flags.DEFINE_bool("save", False, "whether to save results and trajectories")

config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def get_rxn_energy(rxn: str, benchmark: str, df: pd.DataFrame) -> float:
  reactants, products = rxn.replace("-", "").split(">")
  rxn_energy = 0.
  for reactant in reactants.split("+"):
    ratio, system = reactant.split("*")
    rxn_energy -= float(ratio) * df.loc[f"{benchmark}-{system}", "e_total"]
  for product in products.split("+"):
    ratio, system = product.split("*")
    rxn_energy += float(ratio) * df.loc[f"{benchmark}-{system}", "e_total"]
  return rxn_energy


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

    system_done = []
    if FLAGS.benchmark in cfg.save_dir:
      p = Path(cfg.save_dir)
      logging.info(f"resuming {FLAGS.benchmark} benchmark set")
      with cfg.unlocked():
        cfg.uuid = p.name
        cfg.save_dir = str(p.parent)
      system_done = [f.name.split("-")[-1] for f in p.iterdir() if f.is_dir()]
    else:
      with cfg.unlocked():
        cfg.uuid = ",".join([FLAGS.benchmark, cfg.get_core_cfg_str(), cfg.uuid])

    cfg.save()

    systems, _, _ = get_refdata_benchmark_set(FLAGS.benchmark)
    # HACK: fix the xc grad NaN issue
    system_done.append("bh76_h")
    for system in systems:
      if system in system_done:
        logging.info(f"skipping {system}..")
        continue

      with cfg.unlocked():
        cfg.mol_cfg.mol = "-".join([FLAGS.benchmark, system])

      try:
        if FLAGS.run == "direct":
          incore_cgto_direct_opt(cfg, FLAGS.pyscf, FLAGS.basis_optim)
        else:
          raise NotImplementedError
      except Exception as e:
        logging.error(e)

    return

  if FLAGS.run == "direct":
    incore_cgto_direct_opt(cfg, FLAGS.pyscf, FLAGS.basis_optim)

  elif FLAGS.run == "scf":
    incore_cgto_scf(cfg)

  elif FLAGS.run == "pyscf":
    incore_cgto_pyscf_benchmark(cfg)

  elif FLAGS.run == "viz":
    p = Path(cfg.save_dir)
    benchmark = p.name.split(',')[0]
    runs = [f for f in p.iterdir() if f.is_dir()]
    direct_df = pd.DataFrame()
    pyscf_df = pd.DataFrame()
    pyscf_mos = []
    direct_mos = []
    for run in runs:
      direct_df_i = pd.read_csv(run / "direct_opt.csv").iloc[-1:]
      pyscf_df_i = pd.read_csv(run / "pyscf.csv")
      direct_df_i.index = [run.name]
      pyscf_df_i.index = [run.name]
      direct_df = pd.concat([direct_df, direct_df_i])
      pyscf_df = pd.concat([pyscf_df, pyscf_df_i])

      with open(run / "pyscf_mo_coeff.pkl", "rb") as f:
        pyscf_mos.append(pickle.load(f))

      with open(run / "traj.pkl", "rb") as f:
        traj = pickle.load(f)
        direct_mos.append(traj.mo_coeff)

    direct_df = direct_df.rename(columns={
      'Unnamed: 0': 'steps',
    })
    pyscf_df = pyscf_df.rename(columns={
      'Unnamed: 0': 'steps',
    })
    diff_df = (direct_df - pyscf_df) * HARTREE_TO_KCAL_PER_MOL
    diff_df['e_total'].dropna().sort_values().plot(
      kind='bar', label='direct - pyscf (kcal/mol)'
    )
    plt.title(f"{benchmark} benchmark set energy difference")
    plt.ylabel("dE (kcal/mol)")
    plt.plot(
      diff_df.index, [1] * len(diff_df.index), 'r--', label='chemical accuracy'
    )
    plt.legend()
    plt.show()

    # HACK: currently H has nan in xc gradient so cannot be computed
    direct_df.loc['bh76-bh76_h'] = pyscf_df.loc['bh76-bh76_h']

    systems, rxns, ref_energy = get_refdata_benchmark_set(benchmark)
    rxn_df = pd.DataFrame(columns=['direct', 'pyscf', 'ref'])
    for rxn, ref in zip(rxns, ref_energy):
      rxn_e_pyscf = get_rxn_energy(rxn, benchmark, pyscf_df)
      rxn_e_pyscf *= HARTREE_TO_KCAL_PER_MOL
      rxn_e_direct = get_rxn_energy(rxn, benchmark, direct_df)
      rxn_e_direct *= HARTREE_TO_KCAL_PER_MOL
      rxn_df.loc[rxn] = [rxn_e_direct, rxn_e_pyscf, ref]

    rxn_diff_df = pd.DataFrame(columns=['direct', 'pyscf'])
    rxn_diff_df['direct'] = rxn_df['direct'] - rxn_df['ref']
    rxn_diff_df['pyscf'] = rxn_df['pyscf'] - rxn_df['ref']

    rxn_diff_df.sort_values(by='direct').plot(kind='bar')
    plt.title(
      f"{benchmark} benchmark set reaction energy (calculated - reference)"
    )
    plt.ylabel("dE (kcal/mol)")
    plt.show()

    (rxn_df['direct'] - rxn_df['pyscf']).plot(
      kind='bar', label='direct - pyscf (kcal/mol)'
    )
    plt.plot(
      rxn_df.index, [1] * len(rxn_df.index), 'r--', label='chemical accuracy'
    )
    plt.plot(rxn_df.index, [-1] * len(rxn_df.index), 'r--')
    plt.title(
      f"{benchmark} benchmark set reaction energy difference (direct - pyscf)"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
  app.run(main)
