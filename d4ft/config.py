from typing import Literal

from ml_collections import ConfigDict
from pydantic.dataclasses import dataclass

config = dict(validate_assignment=True)


@dataclass(config=config)
class OptimizerConfig:
  """Config for the gradient descent DFT solver"""
  epochs: int = 2000
  lr: float = 1e-2
  lr_decay: Literal["none", "piecewise", "exp"] = "piecewise"
  optimizer: Literal["adam", "sgd"] = "adam"
  rng_seed: int = 137


@dataclass(config=config)
class DFTConfig:
  """Config for DFT routine"""
  rks: bool = True
  """whether to run RKS, i.e. use the same coefficients for both spins"""
  xc_type: str = "lda"
  """name of the xc functional to use"""
  quad_level: int = 1
  """quadrature point level, higher means more points"""
  converge_threshold: float = 1e-3
  """threshold for gradient descent convergence checking"""


@dataclass(config=config)
class MoleculeConfig:
  """Config for molecule"""
  mol_name: str = "o2"
  """name of the molecule"""
  basis: str = "sto-3g"
  """name of the atomic basis set"""


def get_config() -> ConfigDict:
  cfg = ConfigDict()
  cfg.optim_cfg = OptimizerConfig()
  cfg.dft_cfg = DFTConfig()
  cfg.mol_cfg = MoleculeConfig()
  return cfg
