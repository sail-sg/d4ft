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
"""Driver / templates for analytical GTO integration routines."""


def incore_eri_int(mol, n_sto_segs, batch_size=2**25, threshold=1e-8):
  """Compute to eri tensor incore.

  Args:
    batch_size: is tuned for A100
  """
  gto, sto_to_gto = obsa.utils.mol_to_obsa_gto(mol)
  n_gtos = gto.angular.shape[0]
  eri_fn = obsa.electron_repulsion_integral
  s4 = obsa.utils.angular_static_args(*[gto.angular] * 4)

  N = jax.vmap(obsa.utils.normalization_constant)

  n_gtos = gto.angular.shape[0]
  ab_idx, counts_ab = obsa.utils.get_2c_combs(n_gtos)
  Ns = jax.jit(N)(gto.angular, gto.exponent)
  logging.info(f"normalization finished, size: {Ns.shape}")

  abab_idx_count = jnp.hstack([ab_idx, ab_idx, counts_ab[:, None]]).astype(int)

  gto_4c_fn = jax.jit(
    obsa.utils.tensorize_4c_sto(eri_fn, len(abab_idx_count), s4, sto=False),
    static_argnames=["n_segs"]
  )
  eri_abab = gto_4c_fn(gto, Ns, abab_idx_count, None, None)

  logging.info(f"block diag (ab|ab) computed, size: {eri_abab.shape}")

  eri_abcd_sto = 0.
  sto_4c_fn = jax.jit(
    obsa.utils.tensorize_4c_sto(eri_fn, batch_size, s4),
    static_argnames=["n_segs"]
  )

  # TODO: contract diag sto first
  n_2c_idx = len(ab_idx)
  num_idx = obsa.utils.unique_ij(n_2c_idx)
  has_remainder = num_idx % batch_size != 0
  num_batches = num_idx // batch_size + int(has_remainder)
  for i in tqdm(range(num_batches)):
    start = batch_size * i
    end = batch_size * (i + 1)
    slice_size = batch_size
    if i == num_batches - 1 and has_remainder:
      end = num_idx
      slice_size = num_idx - start
    start_idx = obsa.utils.get_triu_ij_from_idx(n_2c_idx, start)
    end_idx = obsa.utils.get_triu_ij_from_idx(n_2c_idx, end)
    abcd_idx_counts = obsa.utils.get_4c_combs_range(
      ab_idx, counts_ab, n_2c_idx, start_idx, end_idx, slice_size
    )
    sto_4c_seg_id_i = obsa.utils.get_sto_segment_id(
      abcd_idx_counts[:, :-1], sto_to_gto, four_center=True
    )
    eri_abcd_i = sto_4c_fn(
      gto, Ns, abcd_idx_counts, sto_4c_seg_id_i, n_sto_segs
    )
    eri_abcd_sto += eri_abcd_i

  return eri_abcd_sto
