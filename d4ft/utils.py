import json
from pathlib import Path
from typing import Union

import black
from ml_collections import ConfigDict


def compose(f, g):
  return lambda *a, **kw: f(g(*a, **kw))


def save_cfg(cfg: ConfigDict, save_path: Union[str, Path]):
  with Path(save_path).open("w") as f:
    json.dump(
      json.loads(cfg.to_json_best_effort()), f, sort_keys=True, indent=2
    )


def load_cfg(save_path: Union[str, Path]) -> ConfigDict:
  with Path(save_path).open("r") as f:
    load_cfg = ConfigDict(json.load(f))
  return load_cfg


def pprint(a):
  return black.format_str(repr(a), mode=black.Mode())
