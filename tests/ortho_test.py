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
import scipy
from absl import logging
from absl.testing import absltest, parameterized

from d4ft.hamiltonian.ortho import sqrt_inv


class OrthoTest(parameterized.TestCase):

  @parameterized.parameters(
    ("eig",),
    ("cholesky",),
    ("svd",),
  )
  def test_sqrt_inv(self, method: str) -> None:
    dim = 3
    A_ = np.random.randn(dim, dim)
    A = A_ @ A_.T  # symmetric

    logging.info(f"{method=}")
    B = sqrt_inv(A, method)
    A_inv = B.T @ B
    self.assertTrue(np.allclose(A_inv @ A, np.eye(dim), atol=1e-4))

  def test_generalized_eigh(self) -> None:
    # generate random fock and ovlp
    A = np.random.randn(2, 5, 5)
    F = A.transpose(0, 2, 1) @ A
    B = np.random.randn(5, 5)
    S = B.T @ B

    S_sqrt_inv = sqrt_inv(S)

    eig, C = map(np.stack, zip(*[scipy.linalg.eigh(F[i], S) for i in range(2)]))

    eig_, P = np.linalg.eigh(S_sqrt_inv @ F @ S_sqrt_inv.T)
    C_ = S_sqrt_inv.T @ P

    self.assertTrue(np.allclose(eig, eig_, atol=1e-6))

    # NOTE: currently the eigenvectors has a sign difference
    self.assertTrue(np.allclose(np.abs(C), np.abs(C_), atol=1e-5))


if __name__ == "__main__":
  absltest.main()
