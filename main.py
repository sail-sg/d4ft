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
"""Entrypoint for D4FT."""
from typing import Any

from absl import app, flags
from ml_collections.config_flags import config_flags

from d4ft.config import D4FTConfig
from d4ft.solver.drivers import (
  incore_cgto_direct_opt_dft, incore_cgto_pyscf_dft_benchmark
)

FLAGS = flags.FLAGS
flags.DEFINE_enum("run", "direct", ["direct", "pyscf"], "which routine to run")
config_flags.DEFINE_config_file(name="config", default="d4ft/config.py")


def main(_: Any) -> None:
  cfg: D4FTConfig = FLAGS.config
  print(cfg)

  if FLAGS.run == "direct":
    incore_cgto_direct_opt_dft(cfg)

  elif FLAGS.run == "pyscf":
    incore_cgto_pyscf_dft_benchmark(cfg)


if __name__ == "__main__":
  app.run(main)
