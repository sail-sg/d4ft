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
"""
energy integrands and integration.
"""

from tqdm import tqdm
import time
from collections import namedtuple
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import logging

from d4ft.functions import decov, distmat, set_diag_zero, wave2density
from d4ft.integral import obara_saika as obsa
from d4ft.integral import quadrature as quad
from d4ft.molecule import Molecule

Energies = namedtuple(
  "Energies", ["e_total", "e_kin", "e_ext", "e_xc", "e_hartree", "e_nuc"]
)

Intors = namedtuple("Intors", ["kin", "ext", "eri", "xc", "precal"])
Precal = namedtuple("Precal", ["kin_d", "ext_d", "eri_d"], defaults=(None,) * 3)


def integrand_exc_lda(mo: Callable):
  r"""LDA with spin.
  https://www.chem.fsu.edu/~deprince/programming_projects/lda/
  Local spin-density approximation
    E_\sigma = 2^(1/3) C \int \rho_\sigma^(4/3) dr
    where C = (3/4)(3/\pi)^(1/3)
  Args:
    mo (Callable): a [3] -> [2, N] function, where N is the number of molecular
    orbitals. mo only takes one argment, which is the coordinate.
  Returns:
  """
  C = -3 / 4 * (3 / jnp.pi)**(1 / 3)
  const = jnp.power(2., 1 / 3) * C
  return lambda x: const * jnp.sum(wave2density(mo, keep_spin=True)(x)**(4 / 3))


def integrand_vxc_lda(ao: Callable, mo_old):
  """
  v_xc = -(3/pi n(r))^(1/3)
  Return:
    [2, N, N] array
  """
  density = wave2density(mo_old)

  def g(n):
    return -(3 / jnp.pi * n)**(1 / 3)

  return lambda r: g(density(r)) * jax.vmap(jnp.outer)(ao(r), ao(r))


def e_nuclear(nuclei):
  """
    Potential energy between atomic nuclears.
  """
  nuclei_loc = nuclei['loc']
  nuclei_charge = nuclei['charge']
  dist_nuc = distmat(nuclei_loc)
  charge_outer = jnp.outer(nuclei_charge, nuclei_charge)
  charge_outer = set_diag_zero(charge_outer)
  return jnp.sum(charge_outer / (dist_nuc + 1e-15)) / 2


def precal_ao_matrix_to_qrmo(params, mat, nocc):
  """Given a whitened AO integral matrix
  <phi_i|O|phi_j> where phi_i are AOs and O is some
  observable, compute the orthogonal MO integral matrix.

  QR decomposition is used to ensure orthogonality

  Args:
    params: a tuple of (mo_params, ao_params)
  """
  mo_params, _ = params
  mo_params = jnp.expand_dims(mo_params, 0)
  mo_params = jnp.repeat(mo_params, 2, 0)  # shape: [2, N, N]

  def transform_by_qrmo_coeff(param, nocc):
    orthogonal, _ = jnp.linalg.qr(param)
    orthogonal *= jnp.expand_dims(nocc, axis=0)
    return jnp.sum(jnp.diagonal(orthogonal.T @ mat @ orthogonal))

  return jnp.sum(jax.vmap(transform_by_qrmo_coeff)(mo_params, nocc))


def precal_ao_matrix(mol: Molecule, batch_size: int, seed: int):
  overlap_decov = decov(mol.ao.overlap())

  batches = quad.utils.make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=1, num_copies=1, seed=seed
  )

  @jax.jit
  def ao_kin_mat_fun(batch):
    kinetic = quad.kinetic_integral(mol.ao, batch, use_jac=False, keepdims=True)
    return overlap_decov @ kinetic @ overlap_decov.T

  @jax.jit
  def ao_ext_mat_fun(batch):
    ext = quad.nuclear_attraction_integral(
      mol.ao, mol.nuclei["loc"], mol.nuclei["charge"], batch, keepdims=True
    )
    return overlap_decov @ ext @ overlap_decov.T

  ao_kin_mat = jnp.zeros([mol.nao, mol.nao])
  ao_ext_mat = jnp.zeros([mol.nao, mol.nao])

  for batch in batches:
    ao_kin_mat += ao_kin_mat_fun(batch)
    ao_ext_mat += ao_ext_mat_fun(batch)

  # TODO: check whether this is needed
  # num_batches = mol.grids.shape[0] // batch_size
  # ao_kin_mat /= num_batches
  # ao_ext_mat /= num_batches

  return ao_kin_mat, ao_ext_mat


def precal_scf(mol: Molecule, batch_size: int, seed: int):
  """
  TODO: benchmark with minibatch_vmap
  """
  batches = quad.utils.make_quadrature_points_batches(
    mol.grids, mol.weights, batch_size, epochs=1, num_copies=1, seed=seed
  )

  nmo = mol.mo.nmo
  diag_one = jnp.ones([2, nmo])
  diag_one = jax.vmap(jnp.diag)(diag_one)

  # NOTE: these ao will have the spin axis
  def ao(r):
    return mol.mo((diag_one, None), r)

  @jax.jit
  def kin_fun(batch):
    return quad.kinetic_integral(ao, batch, use_jac=False, keepdims=True)

  @jax.jit
  def ext_fun(batch):
    return quad.nuclear_attraction_integral(
      ao, mol.nuclei["loc"], mol.nuclei["charge"], batch, keepdims=True
    )

  kin = jnp.zeros([2, nmo, nmo])
  ext = jnp.zeros([2, nmo, nmo])

  for batch in batches:
    kin += kin_fun(batch)
    ext += ext_fun(batch)

  return kin, ext


def calc_fock(
  ao: Callable,
  mo_old: Callable,
  nuclei,
  batch1,
  batch2=None,
  precal_h=None,
):
  """Calculate Fock matrix for SCF."""
  if batch2 is None:
    batch2 = batch1
  if precal_h is None:
    kin = quad.kinetic_integral(ao, batch1, use_jac=False, keepdims=True)
    ext = quad.nuclear_attraction_integral(
      ao, nuclei["loc"], nuclei["charge"], batch1, keepdims=True
    )
  else:
    kin, ext = precal_h
  hartree = quad.electron_repulsion_integral_scf(ao, mo_old, batch1, batch2)
  vxc = quad.utils.quadrature_integral(integrand_vxc_lda(ao, mo_old), batch1)
  return kin + ext + hartree + vxc


def get_xc_intor(mol, xc_type: str = "lda") -> Callable:
  """only support quadrature now"""

  def xc(params, batch1, **args):
    orbitals = lambda r: mol.mo(params, r) * mol.nocc

    if xc_type == 'lda':
      return quad.utils.quadrature_integral(integrand_exc_lda(orbitals), batch1)
    else:
      raise NotImplementedError

  return xc


def get_intor(
  mol,
  batch_size: int,
  seed: int,
  pre_cal: bool = False,
  xc: str = "lda",
  os_scheme: str = "none",
  **kwargs,
) -> Intors:
  if os_scheme != "none":
    use_horizontal = (os_scheme == "horizontal")
    stochastic = (os_scheme in ["uniform", "is"])
    kin, ext, eri, precal = get_os_intor(
      mol, use_horizontal, pre_cal, stochastic, **kwargs
    )
  else:
    kin, ext, eri = get_quadrature_intor(mol, batch_size, seed, pre_cal)
    precal = Precal()
  return Intors(kin, ext, eri, get_xc_intor(mol, xc), precal)


def get_quadrature_intor(
  mol, batch_size: int, seed: int, pre_cal: bool = False
):
  ao_kin_mat, ao_ext_mat = None, None
  if pre_cal:
    logging.info('Preparing for integration...')
    start = time.time()
    ao_kin_mat, ao_ext_mat = precal_ao_matrix(mol, batch_size, seed)
    logging.info(f"Pre-calculation finished. Time: {(time.time()-start):.3f}")

  def kin(params, batch1, *args, **kwargs):
    orbitals = lambda r: mol.mo(params, r) * mol.nocc
    if pre_cal:
      return precal_ao_matrix_to_qrmo(params, ao_kin_mat, mol.nocc)
    return quad.kinetic_integral(orbitals, batch1, use_jac=True)

  def ext(params, batch1, *args, **kwargs):
    orbitals = lambda r: mol.mo(params, r) * mol.nocc
    if pre_cal:
      return precal_ao_matrix_to_qrmo(params, ao_ext_mat, mol.nocc)
    return quad.nuclear_attraction_integral(
      orbitals, mol.nuclei["loc"], mol.nuclei["charge"], batch1
    )

  def eri(params, batch1, batch2, *args, **kwargs):
    orbitals = lambda r: mol.mo(params, r) * mol.nocc
    return quad.electron_repulsion_integral(
      orbitals, batch1=batch1, batch2=batch2
    )

  return kin, ext, eri


# TODO: should this be outside of jit?
def get_obsa_mo(params, mol, gto, sto_to_gto, return_sto_coeff: bool = False):
  mo_coeff = mol.mo.get_mo_coeff(params)
  if len(mo_coeff.shape) == 2:  # restrictied mo
    mo_coeff_spin = jnp.repeat(mo_coeff[None], 2, 0)  # add spin axis
  elif len(mo_coeff.shape) == 3:  # unrestrictied mo
    mo_coeff_spin = mo_coeff
  else:
    raise ValueError
  mo_coeff_spin *= mol.nocc[:, :, None]  # apply spin mask
  mo_coeff_spin = mo_coeff_spin.reshape(-1, mol.mo.nmo)  # flatten
  if return_sto_coeff:
    return mo_coeff_spin

  gto_coeff = np.repeat(
    mo_coeff_spin, np.array(sto_to_gto), axis=-1
  ) * gto.coeff
  mo = obsa.utils.MO(gto.angular, gto.center, gto.exponent, gto_coeff)

  return mo


def get_n_gtos(mol) -> int:
  _, sto_to_gto = obsa.utils.mol_to_obsa_gto(mol)
  return sum(sto_to_gto)


def get_os_intor(
  mol,
  use_horizontal: bool = False,
  pre_cal: bool = False,
  stochastic: bool = False,
  **kwargs,
):
  """
  TODO: more efficient contraction

  Args:
    pre_cal: if True, pre calculate integral tensor and normalizing factor
  """
  gto, sto_to_gto = obsa.utils.mol_to_obsa_gto(mol)

  s2 = obsa.utils.angular_static_args(gto.angular, gto.angular)
  s4 = obsa.utils.angular_static_args(*[gto.angular] * 4)

  kin_fn = partial(obsa.kinetic_integral, use_horizontal=use_horizontal)
  eri_fn = obsa.electron_repulsion_integral

  def ext_fn(a, b, static_args):
    ni = obsa.nuclear_attraction_integral
    return jax.vmap(lambda Z, C: Z * ni(C, a, b, static_args, use_horizontal)
                   )(mol.nuclei["charge"], mol.nuclei["loc"]).sum()

  N = jax.vmap(obsa.utils.normalization_constant)

  precal = Precal()
  if pre_cal:
    logging.info("Precal...")

    n_gtos = gto.angular.shape[0]
    n_stos = len(sto_to_gto)

    Ns = jax.jit(N)(gto.angular, gto.exponent)

    sto_abcd_idx, sto_counts_abcd = obsa.utils.get_4c_combs(n_stos)
    eri_abcd = prescreen_and_precal_4c(mol, len(sto_abcd_idx))

    # logging.info(f"4c precal finished, tensor size: {eri_abcd.shape}")

    # checking with libcint
    # n_4c_idx = obsa.utils.unique_ijkl(n_stos)
    # abcd_idx, counts_abcd = obsa.utils.get_4c_combs_alt(n_gtos)
    # abcd_idx_counts = jnp.hstack([abcd_idx, counts_abcd[..., None]])
    # sto_abcd_idx, sto_counts_abcd = obsa.utils.get_4c_combs_alt(n_stos)
    # sto_4c_seg_id = obsa.utils.get_sto_segment_id_alt(
    #   abcd_idx, sto_to_gto, four_center=True
    # )

    # eri_abcd = jax.jit(
    #   obsa.utils.tensorize_4c_sto(eri_fn, len(abcd_idx), s4, sto=True),
    #   static_argnames=["n_segs"]
    # )(gto, Ns, abcd_idx_counts, sto_4c_seg_id, n_4c_idx)

    # eri_pyscf = mol.pyscf_mol.intor('int2e', aosym='s8')
    # eri_pyscf = mol.pyscf_mol.intor('int2e')
    # eri_pyscf_sym = eri_pyscf[sto_abcd_idx[:, 0], sto_abcd_idx[:, 1],
    #                           sto_abcd_idx[:, 2], sto_abcd_idx[:, 3]]
    # assert np.allclose(
    #   eri_pyscf_sym * sto_counts_abcd * 0.5, eri_abcd, atol=1e-6
    # )

    # 2c normalization
    ab_idx, counts_ab = obsa.utils.get_2c_combs(n_gtos)
    logging.info(f"normalization finished, size: {Ns.shape}")

    sto_ab_idx, sto_counts_ab = obsa.utils.get_2c_combs(n_stos)

    # 2c tensors
    sto_2c_seg_id = obsa.utils.get_sto_segment_id(ab_idx, sto_to_gto)
    n_segs = len(sto_ab_idx)
    kin_ab = jax.jit(
      obsa.utils.tensorize_2c_sto(kin_fn, s2), static_argnames=["n_segs"]
    )(gto, Ns, ab_idx, counts_ab, sto_2c_seg_id, n_segs)
    ext_ab = jax.jit(
      obsa.utils.tensorize_2c_sto(ext_fn, s2), static_argnames=["n_segs"]
    )(gto, Ns, ab_idx, counts_ab, sto_2c_seg_id, n_segs)

    # checking with libcint
    # kin_pyscf = mol.pyscf_mol.intor_symmetric('int1e_kin')
    # kin_pyscf = kin_pyscf[np.triu_indices(n_stos)]
    # assert np.allclose(kin_pyscf * sto_counts_ab, kin_ab, atol=1e-6)

    logging.info(f"2c precal finished, tensor size: {kin_ab.shape}")

    # 4c tensors
    # abcd_idx_count = prescreen(mol)
    # sto_4c_seg_id = obsa.utils.get_sto_segment_id(
    #   abcd_idx_count[:, :-1], sto_to_gto, four_center=True
    # )

    # logging.info("4c index calulated")

    # eri_abcd = jax.jit(
    #   obsa.utils.tensorize_4c_sto(eri_fn, len(abcd_idx_count), s4),
    #   static_argnames=["n_segs"]
    # )(gto, Ns, abcd_idx_count, sto_4c_seg_id, len(sto_abcd_idx))

    # store precomputed tensors and idx, with sparsity
    # NOTE: sparsity changes when changing geometry
    kin_mask = kin_ab != 0
    kin_d = dict(t=kin_ab[kin_mask], idx=sto_ab_idx[kin_mask])
    ext_mask = ext_ab != 0
    ext_d = dict(t=ext_ab[ext_mask], idx=sto_ab_idx[ext_mask])
    eri_mask = eri_abcd != 0
    eri_d = dict(t=eri_abcd[eri_mask], idx=sto_abcd_idx[eri_mask])
    precal = Precal(kin_d=kin_d, ext_d=ext_d, eri_d=eri_d)

  def contract_precal_2c_sto(params, ab_d):
    t_ab, sto_ab_idx = ab_d["t"], ab_d["idx"]
    mo_coeff = get_obsa_mo(params, mol, gto, sto_to_gto, return_sto_coeff=True)
    c_lm = jnp.einsum("il,im->lm", mo_coeff, mo_coeff)
    c_ab = c_lm[sto_ab_idx[:, 0], sto_ab_idx[:, 1]]
    ab = jnp.einsum("k,k->", t_ab, c_ab)
    # coeffs_ab = [mo_coeff[:, sto_ab_idx[:, i]] for i in range(2)]
    # ab = jnp.einsum("k,ik,ik->", t_ab, *coeffs_ab)
    return ab

  def contract_precal_4c_sto(params, abcd_d):
    t_abcd, sto_abcd_idx = abcd_d["t"], abcd_d["idx"]
    mo_coeff = get_obsa_mo(params, mol, gto, sto_to_gto, return_sto_coeff=True)
    c_lm = jnp.einsum("il,im->lm", mo_coeff, mo_coeff)
    c_ab = c_lm[sto_abcd_idx[:, 0], sto_abcd_idx[:, 1]]
    c_cd = c_lm[sto_abcd_idx[:, 2], sto_abcd_idx[:, 3]]
    abcd = jnp.einsum("k,k,k->", t_abcd, c_ab, c_cd)
    # coeffs_abcd = [mo_coeff[:, sto_abcd_idx[:, i]] for i in range(4)]
    # # NOTE: we can compute exact HF exchange here as well
    # abcd = jnp.einsum("k,ik,ik,jk,jk->", t_abcd, *coeffs_abcd)
    return abcd

  def kin(params, precal, *args, **kwargs):
    if pre_cal:
      return contract_precal_2c_sto(params, precal)
    else:
      mo = get_obsa_mo(params, mol, gto, sto_to_gto)
      n_gtos = mo.angular.shape[0]
      return obsa.utils.contraction_2c_sym(kin_fn, n_gtos, s2)(mo)

  def ext(params, precal, *args, **kwargs):
    if pre_cal:
      return contract_precal_2c_sto(params, precal)
    else:
      mo = get_obsa_mo(params, mol, gto, sto_to_gto)
      n_gtos = mo.angular.shape[0]
      return obsa.utils.contraction_2c_sym(ext_fn, n_gtos, s2)(mo)

  def eri(params, idx_count, precal, *args, **kwargs):
    if pre_cal:
      return contract_precal_4c_sto(params, precal)
    else:
      mo = get_obsa_mo(params, mol, gto, sto_to_gto)
      # TODO: organize this
      n_gtos = mo.angular.shape[0]
      if stochastic:
        return obsa.utils.contraction_4c_selected(eri_fn, n_gtos,
                                                  s4)(mo, idx_count)
      else:
        # return obsa.utils.contraction_4c_sym(eri_fn, n_gtos, s4)(mo)
        return obsa.utils.contraction_4c_selected(eri_fn, n_gtos,
                                                  s4)(mo, idx_count)
        # return obsa.utils.contraction_4c_dynamic_prescreen
        # (eri_fn, n_gtos, s4)(
        #   mo
        # )

  return kin, ext, eri, precal


def calc_energy(intors: Intors, nuclei, params, batch1, batch2, idx_count):
  """
  TODO: support for geometry optimization
  """
  e_kin = intors.kin(params, batch1=batch1, precal=intors.precal.kin_d)
  e_ext = intors.ext(params, batch1=batch1, precal=intors.precal.ext_d)
  e_hartree = intors.eri(
    params,
    batch1=batch1,
    batch2=batch2,
    idx_count=idx_count,
    precal=intors.precal.eri_d,
  )
  e_xc = intors.xc(params, batch1=batch1)
  e_nuc = e_nuclear(nuclei)
  e_total = e_kin + e_ext + e_xc + e_hartree + e_nuc
  return e_total, Energies(e_total, e_kin, e_ext, e_xc, e_hartree, e_nuc)


def prescreen_old(mol, params):
  gto, sto_to_gto = obsa.utils.mol_to_obsa_gto(mol)
  mo = get_obsa_mo(params, mol, gto, sto_to_gto)
  n_gtos = mo.angular.shape[0]
  eri_fn = obsa.electron_repulsion_integral
  s4 = obsa.utils.angular_static_args(*[gto.angular] * 4)
  compute_mask, compute_abcd_idx_count = obsa.utils.prescreen_4c(
    eri_fn, n_gtos, s4
  )
  prescreen_mask = compute_mask(mo)
  idx_count = compute_abcd_idx_count(prescreen_mask)
  # account for symmetry
  # schwartz_bound = schwartz_bound[prescreen_mask] * idx_count[:, -1]
  return idx_count, None


def prescreen_and_precal_4c(mol, n_sto_segs, batch_size=2**25, threshold=1e-8):
  """
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

  ###########
  # abcd_idx, counts_abcd = obsa.utils.get_4c_combs(n_gtos)
  # logging.info(f"(ab|cd) index generated, size: {abcd_idx.shape}")

  # abcd_idx_count = jnp.hstack([abcd_idx, counts_abcd[..., None]])
  # num_idx = len(abcd_idx)
  # num_batches = num_idx // batch_size + int(num_idx % batch_size != 0)

  # logging.info(f"batch_size: {batch_size}, num_shards: {num_batches}")

  # abcd_idx = []
  # counts_abcd = []
  # screened_idx = 0
  # n_idx = 0
  # for i in tqdm(range(num_batches)):
  #   sto_4c_seg_id_i = obsa.utils.get_sto_segment_id(
  #     abcd_idx_count[batch_size * i:batch_size * (i + 1), :-1],
  #     sto_to_gto,
  #     four_center=True
  #   )
  #   eri_abcd_i = sto_4c_fn(
  #     gto, Ns, abcd_idx_count[batch_size * i:batch_size * (i + 1)],
  #     sto_4c_seg_id_i, n_sto_segs
  #   )
  #   eri_abcd_sto += eri_abcd_i

  ###########

  # sto_diag_seg_id = obsa.utils.get_sto_segment_id(
  #   abab_idx_count[:, :-1], sto_to_gto, four_center=True
  # )

  # @partial(jax.jit, static_argnames=["n_segs"])
  # def calc_diag_sto(abab, sto_seg_id, n_segs):
  #   return jax.ops.segment_sum(abab, sto_seg_id, n_segs)

  # eri_abcd = calc_diag_sto(eri_abab, sto_diag_seg_id, n_sto_segs)
  # logging.info(f"(ab|ab) contracted to sto, size: {eri_abcd.shape}")

  # n_2c_idx = len(ab_idx)
  # # don't include diag
  # block_triu_idx_dt = tf.data.Dataset.range(n_2c_idx).flat_map(
  #   lambda x: tf.data.Dataset.range(x + 1, n_2c_idx).map(lambda y: (x, y))
  # )
  # dt = block_triu_idx_dt.map(lambda ab, cd: tf.stack([ab, cd]))
  # dt = dt.batch(batch_size)
  # dt = iter(dt)
  # num_idx = obsa.utils.unique_ij(n_2c_idx) - n_2c_idx
  # num_batches = num_idx // batch_size + int(num_idx % batch_size != 0)

  # for _ in tqdm(range(num_batches)):
  #   ab_block_idx = next(dt).numpy()
  #   abcd_idx_counts = obsa.utils.get_4c_combs_new(ab_block_idx, n_gtos)
  #   logging.info(f"get_4c_combs_new")
  #   sto_4c_seg_id_i = obsa.utils.get_sto_segment_id(
  #     abcd_idx_counts[:, :-1], sto_to_gto, four_center=True
  #   )
  #   logging.info(f"get_sto_segment_id")
  #   eri_abcd_i = sto_4c_fn(
  #     gto, Ns, abcd_idx_counts, sto_4c_seg_id_i, n_sto_segs
  #   )
  #   logging.info(f"sto_4c_fn")
  #   eri_abcd_sto += eri_abcd_i

  ###########

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


def prescreen(mol, batch_size=2**18, threshold=1e-8, sto=False):
  """
  TODO: tune the batch_size
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

  if sto:
    n_stos = len(sto_to_gto)
    sto_4c_seg_id = obsa.utils.get_sto_segment_id(
      abab_idx_count[:, :-1], sto_to_gto, four_center=True
    )

    ab_idx, _ = obsa.utils.get_2c_combs(n_stos)
    sto_abcd_idx, _ = obsa.utils.get_4c_combs(n_stos)
    n_2c_idx = len(ab_idx)
    n_4c_idx = len(sto_abcd_idx)
  else:
    sto_abcd_idx = sto_4c_seg_id = None
    n_2c_idx = len(ab_idx)
    n_4c_idx = None

  eri_abab = jax.jit(
    obsa.utils.tensorize_4c_sto(
      eri_fn, len(abab_idx_count), s4, sto=sto, screen=True
    ),
    static_argnames=["n_segs"]
  )(gto, Ns, abab_idx_count, sto_4c_seg_id, n_4c_idx)

  if sto:
    eri_abab = eri_abab[jnp.unique(sto_4c_seg_id, size=n_2c_idx)]

  block_triu_idx_dt = tf.data.Dataset.range(n_2c_idx).flat_map(
    lambda x: tf.data.Dataset.range(x, n_2c_idx).map(lambda y: (x, y))
  )
  dt = block_triu_idx_dt.map(lambda ab, cd: tf.stack([ab, cd]))
  dt = dt.batch(batch_size)
  dt = iter(dt)
  num_idx = obsa.utils.unique_ij(n_2c_idx)
  num_batches = num_idx // batch_size + int(num_idx % batch_size != 0)

  logging.info(f"batch_size: {batch_size}, num_shards: {num_batches}")

  abcd_idx = []
  counts_abcd = []
  screened_idx = 0
  n_idx = 0
  for _ in tqdm(range(num_batches)):
    ab_block_idx = next(dt).numpy()
    offdiag_ab_block = ab_block_idx[:, 0] != ab_block_idx[:, 1]
    counts_ab_block = offdiag_ab_block + jnp.ones(len(ab_block_idx))
    in_block_counts = (
      counts_ab[ab_block_idx[:, 0]] * counts_ab[ab_block_idx[:, 1]]
    )
    between_block_counts = counts_ab_block
    counts_abcd_i = in_block_counts * between_block_counts
    counts_abcd_i = counts_abcd_i.astype(jnp.int32)
    pmask = jnp.sqrt(
      eri_abab[ab_block_idx[:, 0]] * eri_abab[ab_block_idx[:, 1]]
    ) > threshold
    n_idx += pmask.shape[0]
    screened_idx += pmask.sum()

    abcd_idx_i = jnp.hstack(
      [ab_idx[ab_block_idx[:, 0]], ab_idx[ab_block_idx[:, 1]]]
    )[pmask]
    counts_abcd_i = counts_abcd_i[pmask]
    abcd_idx.append(abcd_idx_i)
    counts_abcd.append(counts_abcd_i)

  screen_ratio = screened_idx / n_idx
  logging.info(
    f"sto: {sto}, num unique idx: {n_idx}, "
    f"screen ratio: {screen_ratio*100:.2f}%"
  )

  abcd_idx = np.concatenate(abcd_idx)
  counts_abcd = np.concatenate(counts_abcd)

  if sto:  # get back gto idx
    utr_4c_idx_vmap = jax.vmap(
      lambda ijkl: obsa.utils.utr_4c_idx(n_stos, ijkl), in_axes=0
    )
    screened_sto_idx = utr_4c_idx_vmap(abcd_idx)

    gto_abcd_idx, counts_abcd = obsa.utils.get_4c_combs(n_gtos)
    sto_4c_seg_id = obsa.utils.get_sto_segment_id(
      gto_abcd_idx, sto_to_gto, four_center=True
    )
    mask = np.isin(sto_4c_seg_id, screened_sto_idx)

    abcd_idx = gto_abcd_idx[mask]
    counts_abcd = counts_abcd[mask]

  abcd_idx_count = jnp.hstack([abcd_idx, counts_abcd[:, None]])

  return abcd_idx_count
