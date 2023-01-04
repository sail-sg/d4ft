# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logger for metrics."""

import time
from pathlib import Path

import pandas as pd
from absl import logging


class RunLogger:

  def __init__(self):
    self.data_df = pd.DataFrame()
    self.last_t = 0
    self.start_time = self._time = time.time()

  def reset(self):
    self.last_t = self.data.index[-1]

  def save_as_csv(self, csv_path: str):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    self.data_df.to_csv(csv_path)

  def log_step(self, metrics, t: int):
    step_df = pd.DataFrame([metrics], index=[t])
    # log step time
    now = time.time()
    step_df['time'] = now - self._time
    self._time = now
    self.data_df = pd.concat([self.data_df, step_df])

  def get_segment_summary(self):
    segment_df = self.data_df[self.last_t:]
    self.last_t = self.data_df.index[-1]
    logging.info(f"Iter: {self.last_t}\n{segment_df.mean()}")
    return segment_df

  def log_summary(self):
    logging.info(
      f"Total epochs run: {self.data_df.index[-1]+1}. \n"
      f"Training Time: {(time.time() - self.start_time):.3f}s. \n"
    )
    self.get_segment_summary()
