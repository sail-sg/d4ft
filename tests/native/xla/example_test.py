import jax
import numpy as np
import sys
from absl.testing import absltest
from d4ft.native.xla.custom_call import CustomCallMeta

sys.path.append("tests")
from example import _Example, _ExampleMember, Parent

Example = CustomCallMeta(
  "Example",
  (_Example,),
  {},
)
ExampleMember = CustomCallMeta(
  "ExampleMember",
  (_ExampleMember,),
  {},
)


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

  def test_example(self) -> None:
    e = Example()
    out = e(self.a, self.b)
    np.testing.assert_array_equal(self.a, out)

  def test_example_member(self) -> None:
    p = Parent()
    em = ExampleMember(p)
    out = em(self.a, self.b)
    np.testing.assert_array_equal(self.a, out)


if __name__ == "__main__":
  absltest.main()
