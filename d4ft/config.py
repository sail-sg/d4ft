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

from typing import Literal

from ml_collections import ConfigDict
from pydantic.dataclasses import dataclass

from pydantic.config import ConfigDict as PydanticConfigDict

pydantic_config = PydanticConfigDict({"validate_assignment": True})


@dataclass(config=pydantic_config)
class OptimizerConfig:
  """Config for the gradient descent DFT solver"""
  epochs: int = 2000
  lr: float = 1e-2
  lr_decay: Literal["none", "piecewise"] = "piecewise"
  optimizer: Literal["adam", "sgd"] = "adam"
  rng_seed: int = 137


@dataclass(config=pydantic_config)
class DirectMinimizationConfig:
  """Config for DFT routine"""
  rks: bool = True
  """whether to run RKS, i.e. use the same coefficients for both spins"""
  xc_type: str = "lda_x"
  """name of the xc functional to use"""
  quad_level: int = 1
  """quadrature point level, higher means more points"""
  converge_threshold: float = 1e-4
  """threshold for gradient descent convergence checking"""
  incore: bool = True
  """Whether to store tensors incore when not optimizing basis.
  If false, tensors are computed on the fly."""
  intor: Literal["obsa", "libcint", "quad"] = "obsa"
  """which integration engine to use"""


@dataclass(config=pydantic_config)
class MoleculeConfig:
  """Config for molecule"""
  mol: str = "o2"
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
  geometry_source: Literal["cccdbd", "pubchem"] = "cccdbd"
  """where to query the geometry from."""


class D4FTConfig(ConfigDict):
  optim_cfg: OptimizerConfig
  direct_min_cfg: DirectMinimizationConfig
  mol_cfg: MoleculeConfig

  def __init__(self) -> None:
    super().__init__(
      {
        "optim_cfg": OptimizerConfig(),
        "direct_min_cfg": DirectMinimizationConfig(),
        "mol_cfg": MoleculeConfig(),
      }
    )


def get_config() -> ConfigDict:
  cfg = D4FTConfig()
  return cfg
