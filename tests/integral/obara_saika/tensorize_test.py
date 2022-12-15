import jax

jax.config.update("jax_enable_x64", True)
from absl.testing import absltest
from absl import logging
import numpy as np
import jax.numpy as jnp
from d4ft.integral.obara_saika.utils import (
  tensorize, contraction, angular_static_args
)
from d4ft.integral.obara_saika.electron_repulsion_integral \
  import electron_repulsion_integral
from d4ft.integral.obara_saika.nuclear_attraction_integral \
  import nuclear_attraction_integral
from d4ft.integral.obara_saika.kinetic_integral import kinetic_integral
from d4ft.integral.obara_saika.overlap_integral import overlap_integral
from d4ft.integral.obara_saika.utils import GTO, MO


class _TestTensorize(absltest.TestCase):

  def setUp(self):
    self.a = (np.array([1, 1, 1]), jnp.array([0., 1., 1.]), jnp.array(2.))
    self.b = (np.array([1, 1, 1]), jnp.array([0., 1., 0.]), jnp.array(1.5))
    self.c = (np.array([1, 1, 1]), jnp.array([1., 1., 0.]), jnp.array(1.3))
    self.d = (np.array([1, 1, 1]), jnp.array([0., 0., 0.]), jnp.array(1.2))
    self.abcd = GTO(
      *(jnp.stack(a, axis=0) for a in zip(self.a, self.b, self.c, self.d))
    )
    self.C = jnp.array([1., 1., 1.])
    self.s2 = self._static_args(2)
    self.s4 = self._static_args(4)
    self.coeff = jnp.array([[0.1, 0.2, 0.3, 0.4]])
    self.mo = MO(*self.abcd, self.coeff)

  def _static_args(self, num_centers):
    return angular_static_args(*[self.abcd[0]] * num_centers)

  def _attraction(self, a, b, static_args):
    return nuclear_attraction_integral(self.C, a, b, static_args)

  def test_tensorize_overlap(self):
    overlap_fn = tensorize(overlap_integral, 2, self.s2)
    out = overlap_fn(*[self.abcd] * 2)
    out = jax.jit(overlap_fn)(*[self.abcd] * 2)
    logging.info(f"Overlap: {out.shape}")

  def test_tensorize_kinetic(self):
    kinetic_fn = tensorize(kinetic_integral, 2, self.s2)
    out = kinetic_fn(*[self.abcd] * 2)
    out = jax.jit(kinetic_fn)(*[self.abcd] * 2)
    logging.info(f"Kinetic: {out.shape}")

  def test_tensorize_electron_repulsion(self):
    eri_fn = tensorize(electron_repulsion_integral, 4, self.s4)
    out = eri_fn(*[self.abcd] * 4)
    out = jax.jit(eri_fn)(*[self.abcd] * 4)
    logging.info(f"ERI: {out.shape}")

  def test_tensorize_nuclear_attraction(self):
    nuclear_attraction_fn = tensorize(self._attraction, 2, self.s2)
    out = nuclear_attraction_fn(*[self.abcd] * 2)
    out = jax.jit(nuclear_attraction_fn)(*[self.abcd] * 2)
    logging.info(f"Nuclear attraction: {out.shape}")

  def test_contraction_overlap(self):
    mo = self.mo
    overlap_fn = contraction(overlap_integral, 2, self.s2)
    out = overlap_fn(mo, mo)
    out = jax.jit(overlap_fn)(mo, mo)
    logging.info(f"Overlap: {out}")

  def test_contraction_kinetic(self):
    mo = self.mo
    kinetic_fn = contraction(kinetic_integral, 2, self.s2)
    out = kinetic_fn(mo, mo)
    out = jax.jit(kinetic_fn)(mo, mo)
    logging.info(f"Kinetic: {out}")

  def test_contraction_electron_repulsion(self):
    mo = self.mo
    eri_fn = contraction(electron_repulsion_integral, 4, self.s4)
    out = eri_fn(mo, mo, mo, mo)
    out = jax.jit(eri_fn)(mo, mo, mo, mo)
    logging.info(f"ERI: {out}")

  def test_contraction_nuclear_attraction(self):
    mo = self.mo
    attraction_fn = contraction(self._attraction, 2, self.s2)
    out = attraction_fn(mo, mo)
    out = jax.jit(attraction_fn)(mo, mo)
    logging.info(f"Nuclear attraction: {out}")


if __name__ == "__main__":
  absltest.main()
