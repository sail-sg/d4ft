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
"""
For ease of reference, here we define some term defined in the
original obara-saika paper
"""

import jax.numpy as jnp


def xi(za, zb):
  """z are exponents. Ref eqn.13."""
  return za * zb / zeta(za, zb)


def zeta(za, zb):
  """z are exponents. Ref eqn.14."""
  return za + zb


def rg(ra, rb, rc, za, zb, zc):
  """z are exponents, r are centers. Ref eqn.16."""
  return (za * ra + zb * rb + zc * rc) / (za + zb + zc)


def rp(ra, rb, za, zb):
  """z are exponents, r are centers. Ref eqn.15."""
  return rg(ra, rb, 0.0, za, zb, 0.0)


def rho(zeta, eta):
  """eqn.32"""
  return zeta * eta / (zeta + eta)


def T(rho, pq):
  """eqn.46"""
  return rho * jnp.dot(pq, pq)


def K(z1, z2, r1, r2):
  """eqn.47"""
  d_squared = jnp.dot(r1 - r2, r1 - r2)
  return jnp.sqrt(2) * jnp.pi**(5 / 4) / (z1 + z2) * jnp.exp(
    -z1 * z2 * d_squared / (z1 + z2)
  )


def compute_common_terms(a, b):
  (_, ra, za), (_, rb, zb) = a, b
  assert ra.shape == (3,), "do not pass batch data for this function, use vmap."
  rp_ = rp(ra, rb, za, zb)
  pa = rp_ - ra
  pb = rp_ - rb
  ab = ra - rb
  return zeta(za, zb), rp_, pa, pb, ab, xi(za, zb)


def s_overlap(ra, rb, za, zb):
  """Compute the overlap integral between two s orbitals.

  Ref. TABLE VI, (s||s)

  Args:
    ra, rb: centers
    za, zb: exponents
  """
  ab2 = jnp.linalg.norm(ra - rb, ord=2)**2
  return (jnp.pi / zeta(za, zb))**(3 / 2) * jnp.exp(-xi(za, zb) * ab2)
