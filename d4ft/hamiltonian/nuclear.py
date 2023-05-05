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

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def set_diag_zero(x: Array) -> Array:
  """Set diagonal items to zero."""
  return x.at[jnp.diag_indices(x.shape[0])].set(0)


def e_nuclear(center: Float[Array, "n_atoms 3"],
              charge: Int[Array, "n_atoms"]) -> Float[Array, ""]:
  """Potential energy between atomic nuclears."""
  dist_nuc = jnp.linalg.norm(center - center[:, None], axis=-1)
  charge_outer = jnp.outer(charge, charge)
  charge_outer = set_diag_zero(charge_outer)
  return 0.5 * jnp.sum(charge_outer / (dist_nuc + 1e-15))
