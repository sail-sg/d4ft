# Copyright 2023 Garena Online Private Limited
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


def e_nuclear(center: Float[Array, "n_atoms 3"],
              charge: Int[Array, "n_atoms"]) -> Float[Array, ""]:
  """Potential energy between atomic nuclears."""
  dist_diff = center - center[:, None]
  dist_nuc = jnp.sqrt(jnp.sum(dist_diff**2, axis=-1) + 1e-20)
  dist_nuc = jnp.where(dist_nuc <= 1e-9, 1e20, dist_nuc)
  charge_outer = jnp.outer(charge, charge)
  return 0.5 * jnp.sum(charge_outer / dist_nuc)
