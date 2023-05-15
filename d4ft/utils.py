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

import json
from pathlib import Path
from typing import Union, Any
import jax.numpy as jnp

from ml_collections import ConfigDict


def make_constant_fn(val: Any):
  return lambda *a, **kw: val


def compose(f, g):
  return lambda *a, **kw: f(g(*a, **kw))


def inv_softplus(x):
  return jnp.log(jnp.exp(x) - 1.)


def save_cfg(cfg: ConfigDict, save_path: Union[str, Path]):
  with Path(save_path).open("w") as f:
    json.dump(
      json.loads(cfg.to_json_best_effort()), f, sort_keys=True, indent=2
    )


def load_cfg(save_path: Union[str, Path]) -> ConfigDict:
  with Path(save_path).open("r") as f:
    load_cfg = ConfigDict(json.load(f))
  return load_cfg
