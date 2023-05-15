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
"""Utility code to compute the Boys functions."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from d4ft.integral.obara_saika import boys_table

BoysFuns = jnp.array(boys_table.BoysFuns)
BoysAsympoticConstant = jnp.array(boys_table.BoysAsympoticConstant)


def Boys(m, T):
  return lax.cond(m > 26, lambda: BoysNeville(m, T), lambda: BoysPrecomp(m, T))


def BoysIgamma(m, T):
  """Boys function, eqn.45.

  NOTE: This function is deprecated as it is very slow

  Ref..
  https://www.wolframalpha.com/input?key=&i=int+t%5E%282*m%29*exp%28-T*t%5E2%29+dt+from+0+to+1
  """
  if not jax.config.jax_enable_x64:
    pred = (T < 1e-7)
  else:
    pred = (T < 1e-20)

  def T_is_zero():
    return 1 / (2 * m + 1)

  def T_is_not_zero():
    return (
      1 / 2 * T**(-m - 1 / 2) * jnp.exp(lax.lgamma(m + 1 / 2)) *
      lax.igamma(m + 1 / 2, T)
    )

  return lax.cond(pred, T_is_zero, T_is_not_zero)


def BoysNeville(m, T):
  ngrids = [100, 400, 1600, 6400]
  tdts = [1. / g for g in ngrids]
  ts = [np.linspace(0, 1, g + 1)[:-1] for g in ngrids]
  t = np.concatenate(ts)

  left_endpoint = (m == 0) * 0.5
  # left_endpoint = lax.cond(m == 0, lambda: 0.5, lambda: 0.0)
  right_endpoint = jnp.exp(-T) / 2

  boys_vals = jnp.exp(-T * t * t) * jnp.power(t, 2 * m)

  idx = np.cumsum([0] + ngrids).tolist()
  tresults = [
    (left_endpoint + jnp.sum(boys_vals[idx[i]:idx[i + 1]]) + right_endpoint) *
    tdts[i] for i in range(len(idx) - 1)
  ]

  # tresults = (left_endpoint + tresults + right_endpoint) * tdts

  # Neville polynomial extrapolate them to zero step.
  tresult01 = (-tdts[1]) * (tresults[0] - tresults[1]) / (tdts[0] -
                                                          tdts[1]) + tresults[1]
  tresult12 = (-tdts[2]) * (tresults[1] - tresults[2]) / (tdts[1] -
                                                          tdts[2]) + tresults[2]
  tresult23 = (-tdts[3]) * (tresults[2] - tresults[3]) / (tdts[2] -
                                                          tdts[3]) + tresults[3]
  tresult012 = (-tdts[2]) * (tresult01 -
                             tresult12) / (tdts[0] - tdts[2]) + tresult12
  tresult123 = (-tdts[3]) * (tresult12 -
                             tresult23) / (tdts[1] - tdts[3]) + tresult23
  result = (-tdts[3]) * (tresult012 -
                         tresult123) / (tdts[0] - tdts[3]) + tresult123

  return result


def BoysPrecomp(m, T):
  pred = T > 27.

  def small_T():
    idx0 = (T * 100).astype(int)
    x0 = idx0 / 100.0
    y0 = BoysFuns[m, idx0]
    idx1 = idx0 + 1
    x1 = x0 + 0.01
    y1 = BoysFuns[m, idx1]
    idx2 = idx0 + 2
    x2 = x0 + 0.02
    y2 = BoysFuns[m, idx2]
    idx3 = idx0 + 3
    x3 = x0 + 0.03
    y3 = BoysFuns[m, idx3]
    idx4 = idx0 + 4
    x4 = x0 + 0.04
    y4 = BoysFuns[m, idx4]

    # Neville 5-point interpolation.
    y01 = (T - x1) * (y0 - y1) / (x0 - x1) + y1
    y12 = (T - x2) * (y1 - y2) / (x1 - x2) + y2
    y23 = (T - x3) * (y2 - y3) / (x2 - x3) + y3
    y34 = (T - x4) * (y3 - y4) / (x3 - x4) + y4
    y012 = (T - x2) * (y01 - y12) / (x0 - x2) + y12
    y123 = (T - x3) * (y12 - y23) / (x1 - x3) + y23
    y234 = (T - x4) * (y23 - y34) / (x2 - x4) + y34
    y0123 = (T - x3) * (y012 - y123) / (x0 - x3) + y123
    y1234 = (T - x4) * (y123 - y234) / (x1 - x4) + y234
    y01234 = (T - x4) * (y0123 - y1234) / (x0 - x4) + y1234

    # Downward recursion is not needed.
    return y01234

  def large_T():
    return BoysAsympoticConstant[m] * jnp.power(T, -m - 0.5)

  small = jnp.nan_to_num((1 - pred) * small_T())
  large = jnp.nan_to_num(pred * large_T())
  return large + small
  # return lax.cond(pred, large_T, small_T)
  # return small_T()
