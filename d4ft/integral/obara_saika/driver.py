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
"""Driver / templates for OS GTO integration strategies:
1. incore
2. on the fly
"""
from functools import partial

import jax
import jax.numpy as jnp
from absl import logging
from tqdm import tqdm

from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto import sto_utils, symmetry, tensorization
from d4ft.integral.gto.gto_utils import LCGTO
from d4ft.types import ETensorsIncore


def incore_int(
  lcgto: LCGTO,
  # batch_size: int = 2**25,
  batch_size: int = 2**23,
  prescreen_thresh: float = 1e-8,
  use_horizontal: bool = False,
) -> ETensorsIncore:
  """Compute kin, ext and eri tensor incore.

  ERI tensor is computed in batches.

  Args:
    batch_size: is tuned for A100
  """

  kin_fn = partial(obsa.kinetic_integral, use_horizontal=use_horizontal)
  eri_fn = obsa.electron_repulsion_integral

  def ext_fn(a, b, static_args):
    ni = obsa.nuclear_attraction_integral
    atom_coords = lcgto.primitives.center[
      jnp.cumsum(jnp.array(lcgto.atom_splits)) - 1]
    return jax.vmap(lambda Z, C: Z * ni(C, a, b, static_args, use_horizontal)
                   )(lcgto.charge, atom_coords).sum()

  s2 = obsa.angular_static_args(*[lcgto.primitives.angular] * 2)
  s4 = obsa.angular_static_args(*[lcgto.primitives.angular] * 4)

  # 2c tensors
  ab_idx_counts = symmetry.get_2c_sym_idx(lcgto.n_gtos)
  sto_2c_seg_id = sto_utils.get_sto_segment_id_sym(
    ab_idx_counts, lcgto.cgto_splits
  )
  n_sto_segs_2c = symmetry.unique_ij(lcgto.n_cgtos)
  n_sto_segs_4c = symmetry.unique_ijkl(lcgto.n_cgtos)
  kin_ab = tensorization.tensorize_2c_sto(kin_fn, s2)(
    lcgto, ab_idx_counts, sto_2c_seg_id, n_sto_segs_2c
  )
  ext_ab = tensorization.tensorize_2c_sto(ext_fn, s2)(
    lcgto, ab_idx_counts, sto_2c_seg_id, n_sto_segs_2c
  )
  logging.info(f"2c precal finished, tensor size: {kin_ab.shape}")

  # 4c tensors
  ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
  abab_idx_count = jnp.hstack([ab_idx, ab_idx, counts_ab[:, None]]).astype(int)

  gto_4c_fn = tensorization.tensorize_4c_sto(eri_fn, s4, sto=False)
  sto_4c_fn = tensorization.tensorize_4c_sto(eri_fn, s4)
  # sto_4c_fn = tensorization.tensorize_4c_sto_range(eri_fn, s4)

  eri_abab = gto_4c_fn(lcgto, abab_idx_count, None, None)
  logging.info(f"block diag (ab|ab) computed, size: {eri_abab.shape}")

  eri_abcd_sto = 0.

  # TODO: contract diag sto first
  # TODO: prescreen
  n_2c_idx = len(ab_idx_counts)
  num_idx = symmetry.unique_ij(n_2c_idx)
  has_remainder = num_idx % batch_size != 0
  num_batches = num_idx // batch_size + int(has_remainder)
  for i in tqdm(range(num_batches)):
    start = batch_size * i
    end = batch_size * (i + 1)
    slice_size = batch_size
    if i == num_batches - 1 and has_remainder:
      end = num_idx
      slice_size = num_idx - start
    start_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, start)
    end_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, end)
    abcd_idx_counts = symmetry.get_4c_sym_idx_range(
      ab_idx_counts, n_2c_idx, start_idx, end_idx, slice_size
    )
    sto_4c_seg_id_i = sto_utils.get_sto_segment_id_sym(
      abcd_idx_counts[:, :-1], lcgto.cgto_splits, four_center=True
    )
    eri_abcd_i = sto_4c_fn(
      lcgto, abcd_idx_counts, sto_4c_seg_id_i, n_sto_segs_4c
    )
    # eri_abcd_i = sto_4c_fn(
    #   gtos, ab_idx_counts, n_2c_idx, start_idx, end_idx, slice_size,
    #   n_sto_segs_4c
    # )
    eri_abcd_sto += eri_abcd_i

  return kin_ab, ext_ab, eri_abcd_sto
