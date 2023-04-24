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

from typing import Callable

import jax
import jax.numpy as jnp

from d4ft.functions import decov


class MO:

  def __init__(self, nmo: int, ao: Callable, restricted_mo: bool):
    """Initialize molecular orbital"""
    super().__init__()
    self.ao = ao
    self.nmo = nmo
    self.restricted_mo = restricted_mo

  def init(self, rng_key):
    shape = (
      [self.nmo, self.nmo] if self.restricted_mo else [2, self.nmo, self.nmo]
    )
    mo_params = jax.random.normal(rng_key, shape) / jnp.sqrt(self.nmo)
    return mo_params, self.ao.init(rng_key)

  def __call__(self, params, r, **kwargs):
    """Evaluate the molecular orbital on r.
    Args:
      params: a tuple of (mo_params, ao_params)
        mo_params is expected to be a [N, N] orthogonal array
        if restricted==True, else the shape is [2, N, N]
      r: [3] array
    Return:
      molecular orbitals: [2, N]
    """
    _, ao_params = params
    ao_vec = self.ao(r, ao_params, **kwargs)
    mo_vec = self.get_mo_coeff(params, **kwargs) @ ao_vec
    if self.restricted_mo:  # add spin axis
      mo_vec = jnp.repeat(mo_vec[None], 2, 0)
    return mo_vec

  def get_mo_coeff(self, params, **kwargs):
    mo_params, _ = params
    return mo_params


class MO_qr(MO):
  """Molecular orbital using QR decomposition."""

  def get_mo_coeff(self, params, **kwargs):
    # TODO: ao_param here is not used
    mo_params, ao_params = params
    orthogonal, _ = jnp.linalg.qr(mo_params)  # q is column-orthogal.
    whiten = decov(self.ao.overlap(params=ao_params, **kwargs))
    transpose_axis = (1, 0) if self.restricted_mo else (0, 2, 1)
    orthogonal = jnp.transpose(orthogonal, transpose_axis)
    mo_coeff = orthogonal @ whiten
    return mo_coeff
