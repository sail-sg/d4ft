# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from d4ft.functions import decov


class MO(object):

  def __init__(self, nmo, ao):
    """Initialize molecular orbital"""
    super().__init__()
    self.ao = ao
    self.nmo = nmo

  def init(self, rng_key):
    mo_params = jax.random.normal(rng_key,
                                  [2, self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    # mo_params, _ = jnp.linalg.qr(mo_params)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r, **kwargs):
    """Evaluate the molecular orbital on r.
    Args:
      params: a tuple of (mo_params, ao_params)
              mo_params is expected to be a [2, N, N] orthogonal array
      r: [3] array
    Return:
      molecular orbitals:[2, N]
    """

    mo_params, ao_params = params
    ao_fun_vec = self.ao(r, ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      return param_i @ decov(
        self.ao.overlap(params=ao_params, **kwargs)
      ) @ ao_fun_vec

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)


class MO_pyscf(MO):

  def __call__(self, params, r):
    mo_params, ao_params = params
    ao_fun_vec = self.ao(r, ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      return param_i @ ao_fun_vec

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)


class MO_qr(object):
  """Molecular orbital using QR decomposition."""

  def __init__(self, nmo, ao):
    """Initialize molecular orbital with QR decomposition."""
    super().__init__()
    self.ao = ao
    self.nmo = nmo

  def init(self, rng_key):
    """Initialize the parameter required by this class."""
    mo_params = jax.random.normal(rng_key,
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r, **kwargs):
    """Compute the molecular orbital on r.

    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    """
    mo_params, ao_params = params
    mo_params = jnp.expand_dims(mo_params, 0)
    mo_params = jnp.repeat(mo_params, 2, 0)
    ao_fun_vec = self.ao(r, ao_params, **kwargs)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose() @ decov(
        self.ao.overlap(params=ao_params, **kwargs)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)


class MO_vpf(object):

  def __init__(self, nmo, ao, vpf):
    self.ao = ao
    self.vpf = vpf
    self.nmo = nmo

  def init(self, rng_key):
    keys = jax.random.split(rng_key, 3)
    mo_params = jax.random.normal(keys[0],
                                  [self.nmo, self.nmo]) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(keys[1]), self.vpf.init(keys[2])

  def __call__(self, params, r, **args):
    """Compute the molecular orbital on r.

    R^3 -> R^N. N-body molecular orbital wave functions.
    input: (N: the number of atomic orbitals.)
      |params: N*N
      |r: (3)
    output:
      |molecular orbitals:(2, N)
    """
    mo_params, ao_params, vpf_params = params
    mo_params = jnp.expand_dims(mo_params, 0)
    mo_params = jnp.repeat(mo_params, 2, 0)

    ao_fun_vec = self.ao(self.vpf(vpf_params, r), ao_params)

    def wave_fun_i(param_i, ao_fun_vec):
      orthogonal, _ = jnp.linalg.qr(param_i)  # q is column-orthogal.
      return orthogonal.transpose() @ decov(
        self.ao.overlap(**args)
      ) @ ao_fun_vec  # (self.basis_num)

    def f(param):
      return wave_fun_i(param, ao_fun_vec)

    return jax.vmap(f)(mo_params)
