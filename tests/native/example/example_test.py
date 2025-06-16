import warnings
import os
# Suppress JAX warnings about CUDA
warnings.filterwarnings("ignore", message="An NVIDIA GPU may be present.*")
# Set JAX to not complain about GPU availability
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jaxlib

# Removed print statements that might confuse Bazel
# print(jax.__version__)
# print(jax.devices())
# print(jaxlib.__version__)
import numpy as np
from absl import logging
from absl.testing import absltest

from d4ft.native.xla.custom_call import CustomCallMeta
from tests.native.example.example import Parent, _Example, _ExampleMember

# from jax.interpreters import ad, batching, mlir, xla

Example = CustomCallMeta("Example", (_Example,), {})
example_fn = Example()

# TODO
# def _example_batch_rule(args, axes):
#   return example_fn(args[1:]), axes

# batching.primitive_batchers[example_fn.prim] = _example_batch_rule

ExampleMember = CustomCallMeta("ExampleMember", (_ExampleMember,), {})


class _ExampleTest(absltest.TestCase):

  def setUp(self):
    key = jax.random.PRNGKey(42)
    key, *subkeys = jax.random.split(key, num=3)
    self.a, self.b = [
      jax.random.normal(
        key=subkeys[0],
        shape=[1, 2, 3],
      ),
      jax.random.randint(
        key=subkeys[1],
        shape=[3, 4, 5],
        minval=0,
        maxval=10,
      ),
    ]

    self.a_b, self.b_b = [
      jax.random.normal(
        key=subkeys[0],
        shape=[2, 1, 2, 3],
      ),
      jax.random.randint(
        key=subkeys[1],
        shape=[2, 3, 4, 5],
        minval=0,
        maxval=10,
      ),
    ]

  def test_example(self) -> None:
    logging.info(jax.devices())
    
    # Custom calls only work in JIT mode in JAX 0.6.0+
    out_jit = jax.jit(example_fn)(self.a, self.b)
    logging.info(out_jit)

    # out_vmap = jax.vmap(example_fn)(self.a_b, self.b_b)
    # logging.info(out_vmap)

    # out_grad = jax.grad(e)(self.a, self.b)
    # logging.info(out_grad)
    np.testing.assert_array_equal(self.a, out_jit)

  def test_example_member(self) -> None:
    p = Parent()
    em = ExampleMember(p)
    
    # Custom calls only work in JIT mode in JAX 0.6.0+
    out_jit = jax.jit(em)(self.a, self.b)
    logging.info(out_jit)
    # out_grad = jax.grad(em)(self.a, self.b)
    # logging.info(out_grad)
    np.testing.assert_array_equal(self.a, out_jit)


if __name__ == "__main__":
  absltest.main()
