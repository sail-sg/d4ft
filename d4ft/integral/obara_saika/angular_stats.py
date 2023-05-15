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
"""Utility code to compute statistics of angular momentum over the
input batch of GTOs. We will mark these stats are static so that JIT
can do constant folding during tracing."""

import numpy as np

from d4ft.types import AngularStats


def angular_static_args(*ns) -> AngularStats:
  """Compute static args for angular momentums.

  Args:
    ns: list of angular momentum vectors, where the length
      is assumed to be between 2 to 4. The vectors are indexed
      alphabetically, e.g. for 4 ns we have (na, nb, nc, nd).
      Each array has shape `(3,)` or `(batch, 3)`.
  """

  def min_max_over_batch(i, *ns):
    """min and max angular momentum in each dimension over the batch"""
    if len(ns) < i + 1:
      return None, None
    n = np.array(ns[i])
    return (n.min(0), n.max(0)) if len(n.shape) == 2 else (n, n)

  def max_over_dim(dims, *ns):
    """max angular momentum of some spatial dimensions over the batch"""
    return sum(np.array(n)[..., dims].sum(-1).max() for n in ns)

  # min/max angular mometum for one group, in each axis
  min_a, max_a = min_max_over_batch(0, *ns)
  min_b, max_b = min_max_over_batch(1, *ns)
  min_c, max_c = min_max_over_batch(2, *ns)
  min_d, max_d = min_max_over_batch(3, *ns)
  max_ab = max_a + max_b
  max_cd = None if max_c is None else max_c + max_d
  max_xyz = max_over_dim([0, 1, 2], *ns)
  max_yz = max_over_dim([1, 2], *ns)
  max_z = max_over_dim([2], *ns)

  return AngularStats(
    min_a, min_b, min_c, min_d, max_a, max_b, max_c, max_d, max_ab, max_cd,
    max_xyz, max_yz, max_z
  )
