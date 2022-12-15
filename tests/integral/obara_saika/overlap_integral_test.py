"""Test overlap integral."""

import jax

jax.config.update("jax_enable_x64", True)
from absl.testing import absltest  # noqa: E402
from absl import logging
import jax.numpy as jnp
import numpy as np
from d4ft.integral.obara_saika.overlap_integral import overlap_integral


class _TestOverlapIntegral(absltest.TestCase):

  def setUp(self):
    self.a = (np.array([1, 1, 1]), jnp.array([0., 1., 1.]), jnp.array(2.))
    self.b = (np.array([1, 1, 1]), jnp.array([0., 1., 0.]), jnp.array(1.5))

  def test_vv(self):
    logging.info(overlap_integral(self.a, self.b, vh=False))

  def test_vh(self):
    logging.info(overlap_integral(self.a, self.b, vh=True))


if __name__ == "__main__":
  absltest.main()
