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

from pathlib import Path
from typing import Literal

from ml_collections import ConfigDict
from pydantic.config import ConfigDict as PydanticConfigDict
from pydantic.dataclasses import dataclass

pydantic_config = PydanticConfigDict({"validate_assignment": True})


@dataclass(config=pydantic_config)
class GDConfig:
  """Config for direct minimization with gradient descent solver."""
  lr: float = 1e-2
  """learning rate"""
  lr_decay: Literal["none", "piecewise", "cosine"] = "none"
  """learning rate schedule"""
  optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
  """which optimizer to use"""
  epochs: int = 4000
  """number of updates/iterations"""
  converge_threshold: float = 1e-4
  """threshold for gradient descent convergence checking"""
  hist_len: int = 50
  """number of steps to used for computing standard deviation of totale energy,
  which is used for gradient descent convergence checking"""
  meta_lr: float = 0.03
  """meta learning rate"""
  meta_opt: Literal["none", "adam", "sgd", "rmsprop"] = "none"
  """meta optimizer to use, none to disable"""


@dataclass(config=pydantic_config)
class SCFConfig:
  """Config for self-consistent field solver."""
  momentum: float = 0.5
  """fock matrix update momentum"""
  epochs: int = 100
  """number of updates/iterations"""


@dataclass(config=pydantic_config)
class IntorConfig:
  """Config for Integrations."""
  incore: bool = True
  """Whether to store tensors incore when not optimizing basis.
  If false, tensors are computed on the fly."""
  intor: Literal["obsa", "libcint", "quad"] = "obsa"
  """which integration engine to use"""
  quad_level: int = 1
  """quadrature point level, higher means more points"""


@dataclass(config=pydantic_config)
class MoleculeConfig:
  """Config for molecule"""
  mol: str = "O2"
  """name of the molecule, or the path to the geometry file, which
  specifies the geometry in the format
  <atom_type> <xyz coordinate in angstrom>.
  For example H2:
  H 0.0000 0.0000 0.0000;
  H 0.0000 0.0000 0.7414;"""
  basis: str = "sto-3g"
  """name of the atomic basis set"""
  spin: int = -1
  """number of unpaired electrons. -1 means all electrons are
  maximally paired, so the spin is 0 or 1."""
  charge: int = 0
  """charge multiplicity"""
  geometry_source: Literal["cccdbd", "refdata", "pubchem"] = "cccdbd"
  """where to query the geometry from."""


@dataclass(config=pydantic_config)
class AlgoConfig:
  """Config for Algorithms."""
  algo: Literal["HF", "KS"] = "KS"
  """Which algorithm to use. HF for Hartree-Fock, KS for Kohn-Sham DFT."""
  restricted: bool = False
  """Whether to run restricted calculation, i.e. enforcing symmetry by using the
  same coefficients for both spins"""
  xc_type: str = "lda_x"
  """Name of the xc functional to use. To mix two XC functional, use the
  syntax a*xc_name_1+b*xc_name_2 where a, b are numbers."""
  rng_seed: int = 137
  """PRNG seed"""


class D4FTConfig(ConfigDict):
  algo_cfg: AlgoConfig
  intor_cfg: IntorConfig
  mol_cfg: MoleculeConfig
  gd_cfg: GDConfig
  scf_cfg: SCFConfig
  uuid: str
  save_dir: str

  def __init__(self, config_string: str) -> None:
    super().__init__(
      {
        "algo_cfg": AlgoConfig(),
        "intor_cfg": IntorConfig(),
        "mol_cfg": MoleculeConfig(),
        "gd_cfg": GDConfig(),
        "scf_cfg": SCFConfig(),
        "uuid": "",
        "save_dir": "_exp",
      }
    )

  def validate(self, spin: int, charge: int) -> None:
    if self.algo_cfg.restricted and self.mol_cfg.mol not in ["bh76_h", "h"]:
      assert spin == 0 and charge == 0, \
        "RESTRICTED only supports closed-shell molecules"

  def get_save_dir(self) -> Path:
    return Path(f"{self.save_dir}/{self.uuid}/{self.mol_cfg.mol}")

  def get_core_cfg_str(self) -> str:
    return "+".join([self.mol_cfg.basis, self.algo_cfg.xc_type])

  def save(self):
    save_path = self.get_save_dir().parent
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.txt", "w") as f:
      f.write(str(self))


def get_config(config_string: str = "") -> D4FTConfig:
  """Return the default configurations.

  Args:
    config_string: currently only set the type of algorithm. Available values:
      "gd", "scf".

  NOTE: for distributed setup, might need to move the dataclass definition
  into this function.
  ref. https://github.com/google/ml_collections\
  #config-files-and-pickling-config_files_and_pickling
  """
  cfg = D4FTConfig(config_string)
  return cfg
