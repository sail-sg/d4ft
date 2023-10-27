import jax
import jaxlib

print(jax.__version__)
print(jax.devices())
print(jaxlib.__version__)
import numpy as np
from absl import logging
from absl.testing import absltest
from d4ft.native.xla.custom_call import CustomCallMeta
from tests.native.xla.example import Parent, _Example, _ExampleMember

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
    out = example_fn(self.a, self.b)
    logging.info(out)

    out_jit = jax.jit(example_fn)(self.a, self.b)
    logging.info(out_jit)

    # out_vmap = jax.vmap(example_fn)(self.a_b, self.b_b)
    # logging.info(out_vmap)

    # out_grad = jax.grad(e)(self.a, self.b)
    # logging.info(out_grad)
    np.testing.assert_array_equal(self.a, out)

  def test_example_member(self) -> None:
    p = Parent()
    em = ExampleMember(p)
    out = em(self.a, self.b)
    logging.info(out)
    # out_grad = jax.grad(em)(self.a, self.b)
    # logging.info(out_grad)
    np.testing.assert_array_equal(self.a, out)


if __name__ == "__main__":
  absltest.main()
