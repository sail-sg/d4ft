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
from typing import Literal, Union

from ml_collections import ConfigDict
from pydantic.config import ConfigDict as PydanticConfigDict
from pydantic.dataclasses import dataclass

pydantic_config = PydanticConfigDict({"validate_assignment": True})


@dataclass(config=pydantic_config)
class GDConfig:
  """Config for direct minimization with gradient descent solver."""
  name: Literal["GD"] = "GD"
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
  basis_optim: str = ""
  """whether to enable basis optimization. Format is comma separated list of
  attributes to optimized in the basis set. For example, 'coeff,exp' means
  to optimize the contraction coefficients and exponents of the GTO basis."""


@dataclass(config=pydantic_config)
class SCFConfig:
  """Config for self-consistent field solver."""
  name: Literal["SCF"] = "SCF"
  momentum: float = 0.5
  """fock matrix update momentum"""
  epochs: int = 100
  """number of updates/iterations"""
  converge_threshold: float = 1e-8
  """threshold for gradient descent convergence checking"""
  basis_optim: str = ""
  """whether to enable basis optimization. Format is comma separated list of
  attributes to optimized in the basis set. For example, 'coeff,exp' means
  to optimize the contraction coefficients and exponents of the GTO basis."""


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
  name: Literal["MOL"] = "MOL"
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
class CrystalConfig:
  """Config for crystal"""
  name: Literal["CRY"] = "CRY"
  direct_lattice_dim: str = "1x1x1"
  """Dimension of the direct lattice, i.e. the number of k points
  (crystal momenta) in each spatial direction. Format is N1xN2xN3."""
  reciprocal_lattice_dim: str = "1x1x1"
  """Dimension of the reciprocal lattice, i.e. the number of reciprocal lattice
  vectors in each spatial direction. Format is N1xN2xN3."""
  energy_cutoff: float = 300.
  """kinetic energy (of G points) cutoff for the plane wave basis set.
  Unit is Hartree"""


@dataclass(config=pydantic_config)
class HFConfig:
  """Config for Hartree-Fock theory."""
  name: Literal["HF"] = "HF"
  restricted: bool = False
  """Whether to run restricted calculation, i.e. enforcing symmetry by using the
  same coefficients for both spins"""
  rng_seed: int = 137
  """PRNG seed"""


@dataclass(config=pydantic_config)
class KSDFTConfig:
  """Config for Kohn-Sham Density functional theory."""
  name: Literal["KS"] = "KS"
  xc_type: str = "lda_x"
  """Name of the xc functional to use. To mix two XC functional, use the
  syntax a*xc_name_1+b*xc_name_2 where a, b are numbers."""
  restricted: bool = False
  """Whether to run restricted calculation, i.e. enforcing symmetry by using the
  same coefficients for both spins"""
  rng_seed: int = 137
  """PRNG seed"""


class D4FTConfig(ConfigDict):
  method_cfg: Union[HFConfig, KSDFTConfig]
  """which QC method to use"""
  solver_cfg: Union[GDConfig, SCFConfig]
  """which solver to use"""
  intor_cfg: IntorConfig
  """integration engine config"""
  sys_cfg: Union[MoleculeConfig, CrystalConfig]
  """config for the system to simulate"""
  uuid: str
  save_dir: str

  def __init__(self, config_string: str) -> None:
    method, solver, sys = config_string.split("-")

    if method.lower() == "hf":
      method_cls = HFConfig
    elif method.lower() == "ks":
      method_cls = KSDFTConfig
    else:
      raise ValueError(f"Unknown method {method}")

    if solver.lower() == "gd":
      solver_cls = GDConfig
    elif solver.lower() == "scf":
      solver_cls = SCFConfig
    else:
      raise ValueError(f"Unknown solver {solver}")

    if sys.lower() == "mol":
      sys_cls = MoleculeConfig
    elif sys.lower() == "crystal":
      sys_cls = CrystalConfig
    else:
      raise ValueError(f"Unknown system {sys}")

    super().__init__(
      {
        "method_cfg": method_cls(),
        "solver_cfg": solver_cls(),
        "intor_cfg": IntorConfig(),
        "sys_cfg": sys_cls(),
        "uuid": "",
        "save_dir": "_exp",
      }
    )

  def validate(self, spin: int, charge: int) -> None:
    if self.method_cfg.restricted and self.sys_cfg.mol not in ["bh76_h", "h"]:
      assert spin == 0 and charge == 0, \
        "RESTRICTED only supports closed-shell molecules"

  def get_save_dir(self) -> Path:
    return Path(f"{self.save_dir}/{self.uuid}/{self.sys_cfg.mol}")

  def get_core_cfg_str(self) -> str:
    return "+".join([self.sys_cfg.basis, self.method_cfg.xc_type])

  def save(self):
    save_path = self.get_save_dir().parent
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "config.txt", "w") as f:
      f.write(str(self))


def get_config(config_string: str = "KS-GD-MOL") -> D4FTConfig:
  """Return the default configurations.

  Args:
    config_string: set the method, solver and sys for the D4FTConfig. Format is
    method-solver-sys, and the default is KS-GD-MOL.

  NOTE: for distributed setup, might need to move the dataclass definition
  into this function.
  ref. https://github.com/google/ml_collections\
  #config-files-and-pickling-config_files_and_pickling
  """
  cfg = D4FTConfig(config_string)
  return cfg
