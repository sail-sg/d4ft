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

from __future__ import annotations  # forward declaration

import jax.numpy as jnp
from jaxtyping import Array, Int


def get_occupation_mask(tot_electron: int, size: int, spin: int,
                        charge: int) -> Int[Array, "2 size"]:
  nocc = jnp.zeros([2, size], dtype=int)
  n_up = (tot_electron - charge + spin) // 2
  n_dn = (tot_electron - charge - spin) // 2
  nocc = nocc.at[0, :n_up].set(1)
  nocc = nocc.at[1, :n_dn].set(1)
  return nocc.astype(int)
