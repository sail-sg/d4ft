#!/usr/bin/env python3
import jax

jax.config.update("jax_enable_x64", True)
from absl.testing import absltest
import numpy as np
from d4ft.integral.obara_saika.overlap_integral import overlap_integral
from d4ft.integral.obara_saika.kinetic_integral import kinetic_integral
from d4ft.integral.obara_saika.nuclear_attraction_integral \
  import nuclear_attraction_integral
from d4ft.integral.obara_saika.electron_repulsion_integral \
  import electron_repulsion_integral
from obsa.obara_saika import get_overlap, get_kinetic, get_nuclear, get_coulomb


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
      o2 = overlap_integral(a, b, vh=False)
      np.testing.assert_allclose(o1, o2)

  def test_overlap_vh(self):
    for ra, rb, _, _, za, zb, _, _, na, nb, _, _, _ in zip(*self.data):
      a, b = (na, ra, za), (nb, rb, zb)
      o1 = get_overlap(za, zb, ra, rb, na.tolist() + nb.tolist())
      o2 = overlap_integral(a, b, vh=True)
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
      e2 = electron_repulsion_integral(a, b, c, d)
      np.testing.assert_allclose(float(e1), float(e2))


if __name__ == "__main__":
  absltest.main()