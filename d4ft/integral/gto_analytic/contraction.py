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
from d4ft.integral.gto_analytic.gto_utils import normalization_constant


def contraction_2c_sym(f, n_gtos, static_args):
  """2c contraction with 2-fold symmetries."""

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0))

  N = jax.vmap(normalization_constant)

  ab_idx, counts_ab = get_2c_combs(n_gtos)

  def g(mo: MO):
    Ns = N(mo.angular, mo.exponent)
    gtos_ab = [
      GTO(*map(lambda gto_param: gto_param[ab_idx[:, i]], mo[:3]))
      for i in range(2)
    ]
    N_ab = Ns[ab_idx].prod(-1) * counts_ab
    t_ab = vmap_f(*gtos_ab)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)
    c_ab = c_lm[ab_idx[:, 0], ab_idx[:, 1]]
    ab = jnp.einsum("k,k,k->", t_ab, N_ab, c_ab)
    return ab

  return g


def contraction_4c_selected(f, n_gtos, static_args):
  """4c centers contraction with provided index set.
  Used together with precomputed prescreen or sampled index set.
  """

  def _f(*args: GTO):
    return f(*args, static_args=static_args)

  vmap_f = jax.vmap(_f, in_axes=(0, 0, 0, 0))

  N = jax.vmap(normalization_constant)

  def g(mo: MO, idx_count):
    Ns = N(mo.angular, mo.exponent)
    c_lm = jnp.einsum("il,im->lm", mo.coeff, mo.coeff)

    abcd_idx = idx_count[:, :4]
    counts_abcd_i = idx_count[:, -1]
    gtos_abcd = [
      GTO(*map(lambda gto_param: gto_param[abcd_idx[:, i]], mo[:3]))
      for i in range(4)
    ]
    N_abcd = Ns[abcd_idx].prod(-1) * counts_abcd_i
    t_abcd = vmap_f(*gtos_abcd)
    c_ab = c_lm[abcd_idx[:, 0], abcd_idx[:, 1]]
    c_cd = c_lm[abcd_idx[:, 2], abcd_idx[:, 3]]
    abcd = jnp.einsum("k,k,k,k->", t_abcd, N_abcd, c_ab, c_cd)
    return abcd

  return g
