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
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from d4ft.integral.obara_saika.nuclear_attraction_integral import (
  nuclear_attraction_integral,
)


class _TestNuclearAttractionIntegral(absltest.TestCase):

  def setUp(self):
    self.a = (np.array([1, 1, 1]), jnp.array([0., 1., 1.]), jnp.array(2.))
    self.b = (np.array([1, 1, 1]), jnp.array([0., 1., 0.]), jnp.array(1.5))
    self.C = jnp.array([1., 1., 1.])

  def test_vv(self):
    print(
      nuclear_attraction_integral(
        self.C,
        self.a,
        self.b,
        use_horizontal=False,
      )
    )

  def test_vh(self):
    print(
      nuclear_attraction_integral(
        self.C,
        self.a,
        self.b,
        use_horizontal=True,
      )
    )


if __name__ == "__main__":
  absltest.main()
