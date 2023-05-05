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
