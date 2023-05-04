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
"""2C and 4C contraction with symmetry"""

import jax
import jax.numpy as jnp

from d4ft.integral.gto.gto_utils import LCGTO


def contraction_2c_sym(f, n_gtos, static_args):
  """4c centers contraction with provided index set."""

  def f_curry(*args: LCGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0))

  ab_idx, counts_ab = get_2c_combs(n_gtos)

  def contract_fn(mo: LCGTO):
    """assuming mo are normalized"""
    Ns = mo.N
    gtos_ab = [
      LCGTO(*map(lambda gto_param: gto_param[ab_idx[:, i]], mo[:3]))
      for i in range(2)
    ]
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*gtos_ab)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    c_ab = c_lm[ab_idx[:, 0], ab_idx[:, 1]]
    ab = jnp.einsum("k,k,k->", t_ab, N_ab, c_ab)
    return ab

  return contract_fn


def contraction_4c(f, static_args):
  """4c centers contraction with provided index set.
  Used together with precomputed prescreen or sampled index set.
  """

  def f_curry(*args: LCGTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(f_curry, in_axes=(0, 0, 0, 0))

  def contract_fn(mo: LCGTO, idx_count):
    """assuming mo are normalized"""
    Ns = mo.N
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)

    abcd_idx = idx_count[:, :4]
    counts_abcd_i = idx_count[:, -1]
    gtos_abcd = [
      LCGTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    t_abcd = vmap_f(*gtos_abcd)
    c_ab = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
    c_cd = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
    abcd = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab, c_cd)
    return abcd

  return contract_fn
