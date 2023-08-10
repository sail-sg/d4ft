from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
from absl import logging

REFDATA_URL = "https://raw.githubusercontent.com/aoterodelaroza/refdata/master"


def get_rxn_str(df_i: pd.DataFrame) -> str:
  return "+".join(
    [f"{abs(ratio)}*{name}" for ratio, name in zip(df_i.types, df_i.names)]
  )


def get_refdata_benchmark_set(
  name: str = "bh76"
) -> Tuple[List[str], List[str], np.ndarray]:
  """Reads a refdata din file, returns:
  1. the systems in the benchmark set
  2. the reactions in the benchmark set
  3. the reference energies in the benchmark set
  """
  url = f"{REFDATA_URL}/10_din/{name}.din"
  res = requests.get(url)
  if res.status_code == 404:
    raise ValueError(f"no benchmark set found for {name}")
  lines = res.content.decode("utf-8").split("\n")
  lines = [l for l in lines if l != "" and l[0] != "#"]
  rxn_df = pd.DataFrame(dict(types=map(float, lines[::2]), names=lines[1::2]))
  ref_energy = rxn_df.loc[rxn_df.types == 0]
  reactants = rxn_df.loc[rxn_df.types < 0]
  products = rxn_df.loc[rxn_df.types > 0]

  systems = pd.concat([reactants.names, products.names]).unique()
  rxns = []
  n_rxns = len(ref_energy.index)
  for idx in range(n_rxns):
    start = 0 if idx == 0 else ref_energy.index[idx - 1]
    end = ref_energy.index[idx]
    reactant_str = get_rxn_str(reactants.loc[start:end])
    product_str = get_rxn_str(products.loc[start:end])
    rxn_str = f"{reactant_str}->{product_str}"
    rxns.append(rxn_str)

  ref_energy = ref_energy.names.astype(float).values
  return systems, rxns, ref_energy


def get_refdata_geometry(name: str) -> Tuple[str, int, int]:
  logging.info("loading geometry from refdata")
  set_name, sys_name = name.split("-")
  url = f"{REFDATA_URL}/20_{set_name}/{sys_name}.xyz"
  res = requests.get(url)
  if res.status_code == 404:
    raise ValueError(f"no geometry found for {name}")
  lines = res.content.decode("utf-8").split("\n")
  n_atoms = int(lines[0].strip())
  assert n_atoms == len(lines) - 3
  charge, spin_p1 = map(int, lines[1].strip().split(" "))
  spin = spin_p1 - 1
  geometry = "\n".join(lines[2:])
  return geometry, charge, spin
