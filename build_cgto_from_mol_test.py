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
from typing import Any

from absl import app, flags, logging
from jax.config import config
from ml_collections.config_flags import config_flags

from d4ft.config import D4FTConfig
from d4ft.constants import HARTREE_TO_KCALMOL
from d4ft.integral.gto.cgto import CGTO
from d4ft.system.mol import Mol, get_pyscf_mol
from pyscf import gto

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "run", "direct", ["direct", "scf", "pyscf", "reaction"],
  "which routine to run"
)
flags.DEFINE_string("reaction", "hf_h_hfhts", "the reaction to run")
flags.DEFINE_bool("use_f64", False, "whether to use float64")

config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_: Any) -> None:
  config.update("jax_enable_x64", FLAGS.use_f64)

  cfg: D4FTConfig = FLAGS.config
  #print(cfg)
  #print(gto.load('ccpvdz', 'O'))
  #print(gto.uncontract(gto.load('ccpvdz', 'O')))
  #print(gto.contract(gto.uncontract(gto.load('ccpvdz', 'O'))))
  #pyscf_mol = get_pyscf_mol(
  #  cfg.mol_cfg.mol, cfg.mol_cfg.basis, cfg.mol_cfg.spin, cfg.mol_cfg.charge,
  #  cfg.mol_cfg.geometry_source
  #)
  #print(gto.format_basis({"O":"cc-pvdz"}))
  #print(pyscf_mol.bas_angular(9))
  import pyscf
  pyscf_mol_cc = gto.M(atom='Kr 0 0 0', basis='cc-pvdz',cart=True,spin=0)
  pyscf_mol_3g = gto.M(atom='Zn 0 0 0', basis='sto-3g',cart=True,spin=0)
  
  #breakpoint()
  pyscf_mol = pyscf_mol_cc
  mol = Mol.from_pyscf_mol(pyscf_mol)
  #print(mol.basis)
  cfg.validate(mol.spin, mol.charge)
  cgto_cart = CGTO.from_mol(mol)
  cgto_sph = CGTO.from_cart(cgto_cart)
  #print(cgto.N_cn)
  #print(cgto.N)
  #print(cgto.N_cn)
  #print(cgto.nao)
  # linear combination d shell
  print("pyscf basis:",pyscf_mol._basis)
  #print("pyscf env:",pyscf_mol._env)
  import numpy
  coords_pyscf = numpy.random.random((1,3))
  ao_value_pyscf = pyscf_mol.eval_gto("GTOval_sph", coords_pyscf)
  ao_value_cgto = cgto_sph.eval(coords_pyscf)
  #print(coords_pyscf)
  print(ao_value_pyscf)
  print(ao_value_cgto)
  print(ao_value_pyscf[0]-ao_value_cgto)
  #print(cgto.nao)





if __name__ == "__main__":
  app.run(main)
