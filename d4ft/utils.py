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

import json
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterable, Type, Union

import einops
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Num
from ml_collections import ConfigDict

from d4ft.types import RDM1, MoCoeff


def complex_norm_square(x):
  return jnp.real(jnp.conj(x) * x)


def make_constant_fn(val: Any) -> Callable:
  return lambda *_, **__: val


def compose(f: Callable, g: Callable) -> Callable:
  return lambda *a, **kw: f(g(*a, **kw))


def inv_softplus(x: Num[Array, "*s"]) -> Num[Array, "*s"]:
  return jnp.log(jnp.exp(x) - 1.)


def save_cfg(cfg: ConfigDict, save_path: Union[str, Path]) -> None:
  with Path(save_path).open("w") as f:
    json.dump(
      json.loads(cfg.to_json_best_effort()), f, sort_keys=True, indent=2
    )


def load_cfg(save_path: Union[str, Path]) -> ConfigDict:
  with Path(save_path).open("r") as f:
    load_cfg = ConfigDict(json.load(f))
  return load_cfg


def to_3dvec(val: Any, dtype: Type) -> Num[np.ndarray, "3"]:
  if isinstance(val, Num[np.ndarray, "3"]):  # already a 3d vector
    return val
  elif isinstance(val, Iterable) and len(val) == 3:  # list of 3 numbers
    return np.array(val, dtype=dtype)
  elif isinstance(val, Number):  # single number
    return np.ones(3, dtype=dtype) * np.array(val, dtype=dtype)
  else:
    raise TypeError("input should be a scalar, Iterable or np.array.")


def vmap_3D_lattice(func: Callable) -> Callable:
  """vmap a funtion in_shape->out_shape to 3D lattice, i.e. transform it to
  a function (x_dim y_dim z_dim in_shape)->(x_dim y_dim z_dim out_shape)"""
  return jax.vmap(jax.vmap(jax.vmap(func)))


def get_rdm1(mo_coeff: MoCoeff) -> RDM1:
  """Calculate the 1-reduced density matrix (1-RDM) from MO coefficients."""
  # return mo_coeff.transpose(0, 2, 1) @ mo_coeff
  # return einops.rearrange(mo_coeff, "spin nmo nao -> spin nao nmo") @ mo_coeff
  return einops.einsum(
    mo_coeff, mo_coeff, "spin mo ao_i, spin mo ao_j -> spin ao_i ao_j"
  )
