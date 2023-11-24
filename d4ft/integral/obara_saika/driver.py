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
"""Driver / templates for OS GTO integration strategies:
1. incore
2. on the fly
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from absl import logging
from tqdm import tqdm

from d4ft.integral import obara_saika as obsa
from d4ft.integral.gto import symmetry, tensorization
from d4ft.integral.gto.cgto import CGTO
from d4ft.types import AngularStats, CGTOSymTensorIncore, Tensor2C, Tensor4C


class CGTOSymTensorFns(NamedTuple):
  """Functions that maps CGTO to symmetry reduced ovlp, kin, ext and eri
  tensor."""
  ovlp_ab_fn: Callable[[CGTO], Tensor2C]
  """Maps CGTO to overlap tensor."""
  kin_ab_fn: Callable[[CGTO], Tensor2C]
  """Maps CGTO to kinetic tensor."""
  ext_ab_fn: Callable[[CGTO], Tensor2C]
  """Maps CGTO to external tensor."""
  eri_abcd_fn: Callable[[CGTO], Tensor4C]
  """Maps CGTO to eri tensor."""

  def get_incore_tensors(self, cgto: CGTO) -> CGTOSymTensorIncore:
    return CGTOSymTensorIncore(
      ovlp_ab=self.ovlp_ab_fn(cgto),
      kin_ab=self.kin_ab_fn(cgto),
      ext_ab=self.ext_ab_fn(cgto),
      eri_abcd=self.eri_abcd_fn(cgto),
    )


def get_cgto_sym_tensor_fns(
  cgto: CGTO,
  s2: AngularStats,
  s4: AngularStats,
  # batch_size: int = 2**25,
  batch_size: int = 2**23,
  prescreen_thresh: float = 1e-8,
  use_horizontal: bool = False,
) -> CGTOSymTensorFns:
  """Compute ovlp, kin, ext and eri tensor with CGTO basis in symmetry reduced
  form incore.

  ERI tensor is computed in batches.

  Args:
    batch_size: is tuned for A100
  """
  ovlp_fn = partial(obsa.overlap_integral, use_horizontal=use_horizontal)
  kin_fn = partial(obsa.kinetic_integral, use_horizontal=use_horizontal)
  eri_fn = obsa.electron_repulsion_integral

  def ext_fn(a, b, static_args):
    ni = obsa.nuclear_attraction_integral
    atom_coords = cgto.pgto.center[jnp.cumsum(jnp.array(cgto.atom_splits)) - 1]
    return jax.vmap(lambda Z, C: Z * ni(C, a, b, static_args, use_horizontal)
                   )(cgto.charge, atom_coords).sum()

  # 2c tensors
  ab_idx_counts = symmetry.get_2c_sym_idx(cgto.n_pgtos)
  cgto_2c_seg_id = symmetry.get_cgto_segment_id_sym(
    ab_idx_counts, cgto.cgto_splits
  )
  n_cgto_segs_2c = symmetry.num_unique_ij(cgto.n_cgtos)
  n_cgto_segs_4c = symmetry.num_unique_ijkl(cgto.n_cgtos)
  ovlp_ab_fn = lambda cgto_: tensorization.tensorize_2c_cgto(ovlp_fn, s2)(
    cgto_, ab_idx_counts, cgto_2c_seg_id, n_cgto_segs_2c
  )
  kin_ab_fn = lambda cgto_: tensorization.tensorize_2c_cgto(kin_fn, s2)(
    cgto_, ab_idx_counts, cgto_2c_seg_id, n_cgto_segs_2c
  )
  ext_ab_fn = lambda cgto_: tensorization.tensorize_2c_cgto(ext_fn, s2)(
    cgto_, ab_idx_counts, cgto_2c_seg_id, n_cgto_segs_2c
  )
  n_2c_idx = len(ab_idx_counts)
  logging.info(f"2c tensor size: {n_2c_idx}")

  # 4c tensors
  # ab_idx, counts_ab = ab_idx_counts[:, :2], ab_idx_counts[:, 2]
  # abab_idx_count = jnp.hstack([ab_idx, ab_idx,
  # counts_ab[:, None]]).astype(int)

  # pgto_4c_fn = tensorization.tensorize_4c_cgto(eri_fn, s4, cgto=False)
  # cgto_4c_fn = tensorization.tensorize_4c_cgto(eri_fn, s4)
  cgto_4c_fn = tensorization.tensorize_4c_cgto_cuda(s4)
  # cgto_4c_fn = tensorization.tensorize_4c_cgto_range(eri_fn, s4)

  # NOTE: these are only needed for prescreening
  # eri_abab = pgto_4c_fn(cgto, abab_idx_count, None, None)
  # logging.info(f"block diag (ab|ab) computed, size: {eri_abab.shape}")

  # TODO: contract diag sto first
  # TODO: prescreen. This requires getting index of off diagonal elements
  num_4c_idx = symmetry.num_unique_ij(n_2c_idx)
  has_remainder = num_4c_idx % batch_size != 0
  num_batches = num_4c_idx // batch_size + int(has_remainder)

  logging.info(f"4c tensor size: {num_4c_idx}, num batches: {num_batches}")

  def eri_abcd_fn(cgto: CGTO) -> Tensor4C:
    eri_abcd_cgto = jnp.zeros(1)

    for i in tqdm(range(num_batches)):
      start = batch_size * i
      end = batch_size * (i + 1)
      slice_size = batch_size
      if i == num_batches - 1 and has_remainder:
        end = num_4c_idx
        slice_size = num_4c_idx - start
      start_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, start)
      end_idx = symmetry.get_triu_ij_from_idx(n_2c_idx, end)
      abcd_idx_counts = symmetry.get_4c_sym_idx_range(
        ab_idx_counts, n_2c_idx, start_idx, end_idx, slice_size
      )
      cgto_4c_seg_id_i = symmetry.get_cgto_segment_id_sym(
        abcd_idx_counts[:, :-1], cgto.cgto_splits, four_center=True
      )
      eri_abcd_i = cgto_4c_fn(
        cgto, abcd_idx_counts, cgto_4c_seg_id_i, n_cgto_segs_4c
      )
      eri_abcd_cgto += eri_abcd_i
    return eri_abcd_cgto

  return CGTOSymTensorFns(ovlp_ab_fn, kin_ab_fn, ext_ab_fn, eri_abcd_fn)
