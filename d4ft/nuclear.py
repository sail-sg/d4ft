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

# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute nuclear repulsion energy."""

import jax
import jax.numpy as jnp


def euclidean_distance(x, y):
  """Euclidean distance."""
  return jnp.sqrt(jnp.sum((x - y)**2 + 1e-18))


def distmat(x, y=None):
  """Distance matrix."""
  if y is None:
    y = x
  return jax.vmap(
    lambda x1: jax.vmap(lambda y1: euclidean_distance(x1, y1))(y)
  )(
    x
  )


def set_diag_zero(x):
  """Set diagonal items to zero."""
  return x.at[jnp.diag_indices(x.shape[0])].set(0)


def e_nuclear(nuclei):
  """Potential energy between atomic nuclears."""
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2
