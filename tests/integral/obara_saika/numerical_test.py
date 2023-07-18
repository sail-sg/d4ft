#!/usr/bin/env python3
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

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from absl.testing import absltest
from obsa.obara_saika import get_coulomb, get_kinetic, get_nuclear, get_overlap

from d4ft.integral.obara_saika.boys import Boys, BoysIgamma
from d4ft.integral.obara_saika.electron_repulsion_integral import (
  electron_repulsion_integral,
)
from d4ft.integral.obara_saika.kinetic_integral import kinetic_integral
from d4ft.integral.obara_saika.nuclear_attraction_integral import (
  nuclear_attraction_integral,
)
from d4ft.integral.obara_saika.overlap_integral import overlap_integral


class _TestNumericalCorrectness(absltest.TestCase):

  def setUp(self):
    self.data = self._random_gto_pairs(10)

  def _random_gto_pairs(self, batch):
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 13)
    ra = jax.random.normal(keys[0], (batch, 3))
    rb = jax.random.normal(keys[1], (batch, 3))
    rc = jax.random.normal(keys[2], (batch, 3))
    rd = jax.random.normal(keys[3], (batch, 3))
    za = jax.random.uniform(keys[4], (batch,), minval=1., maxval=10.)
    zb = jax.random.uniform(keys[5], (batch,), minval=1., maxval=10.)
    zc = jax.random.uniform(keys[6], (batch,), minval=1., maxval=10.)
    zd = jax.random.uniform(keys[7], (batch,), minval=1., maxval=10.)
    na = jax.random.randint(keys[8], (batch, 3), minval=0, maxval=2)
    nb = jax.random.randint(keys[9], (batch, 3), minval=0, maxval=2)
    nc = jax.random.randint(keys[10], (batch, 3), minval=0, maxval=2)
    nd = jax.random.randint(keys[11], (batch, 3), minval=0, maxval=2)
    C = jax.random.normal(keys[12], (batch, 3))
    return list(
      map(np.array, [ra, rb, rc, rd, za, zb, zc, zd, na, nb, nc, nd, C])
    )

  def test_overlap(self):
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*self.data):
      a, b = (na, ra, za), (nb, rb, zb)
      o1 = get_overlap(za, zb, ra, rb, na.tolist() + nb.tolist())
      o2 = overlap_integral(a, b, use_horizontal=False)
      np.testing.assert_allclose(o1, o2)

  def test_overlap_vh(self):
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*self.data):
      a, b = (na, ra, za), (nb, rb, zb)
      o1 = get_overlap(za, zb, ra, rb, na.tolist() + nb.tolist())
      o2 = overlap_integral(a, b, use_horizontal=True)
      np.testing.assert_allclose(o1, o2)

  def test_kinetic(self):
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*self.data):
      a, b = (na, ra, za), (nb, rb, zb)
      k1 = get_kinetic(za, zb, ra, rb, na.tolist() + nb.tolist())
      k2 = kinetic_integral(a, b)
      np.testing.assert_allclose(k1, k2)

  def test_nuclear_attraction(self):
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, C in zip(*self.data):
      a, b = (na, ra, za), (nb, rb, zb)
      n1 = get_nuclear(za, zb, ra, rb, C, na.tolist() + nb.tolist())
      n2 = nuclear_attraction_integral(C, a, b)
      np.testing.assert_allclose(float(n1), float(n2))

  def test_electron_repulsion(self):
    for ra, rb, rc, rd, za, zb, zc, zd, na, nb, nc, nd, _ in zip(*self.data):
      a, b, c, d = (na, ra, za), (nb, rb, zb), (nc, rc, zc), (nd, rd, zd)
      e1 = get_coulomb(
        za, zb, zc, zd, ra, rb, rc, rd,
        na.tolist() + nb.tolist() + nc.tolist() + nd.tolist()
      )
      # *2 as d4ft version already accounts for the 1/2 prefactor
      e2 = 2 * electron_repulsion_integral(a, b, c, d)
      np.testing.assert_allclose(float(e1), float(e2))

  def test_boys(self):
    batch = 10

    key = jax.random.PRNGKey(137)

    M = 5
    T = np.abs(jax.random.normal(key, (batch,)))

    boys_igamma_fn = jax.vmap(
      jax.vmap(BoysIgamma, in_axes=(0, None)), in_axes=(None, 0)
    )
    boys_igamma_grad = jax.vmap(
      jax.vmap(jax.grad(BoysIgamma, argnums=1), in_axes=(0, None)),
      in_axes=(None, 0)
    )
    boys_fast_fn = jax.vmap(
      jax.vmap(Boys, in_axes=(0, None)), in_axes=(None, 0)
    )
    boys_fast_grad = jax.vmap(
      jax.vmap(jax.grad(Boys, argnums=1), in_axes=(0, None)), in_axes=(None, 0)
    )

    np.testing.assert_allclose(
      boys_igamma_fn(np.arange(M), T), boys_fast_fn(np.arange(M), T)
    )

    np.testing.assert_allclose(
      boys_igamma_grad(np.arange(M), T), boys_fast_grad(np.arange(M), T)
    )

    M = 30
    T = np.abs(jax.random.normal(key, (batch,)))

    np.testing.assert_allclose(
      boys_igamma_fn(np.arange(M), T), boys_fast_fn(np.arange(M), T)
    )

    np.testing.assert_allclose(
      boys_igamma_grad(np.arange(M), T), boys_fast_grad(np.arange(M), T)
    )


if __name__ == "__main__":
  absltest.main()
