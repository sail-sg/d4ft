"""A profiling tool to trace python and jax."""
import os.path as osp

import jax
from viztracer import VizTracer


class PyTracer:
  """Tracks python and jax(optional)."""

  def __init__(self, folder: str, name: str, with_jax: bool = False) -> None:
    """Initial method."""
    super().__init__()
    self._name = name
    self._folder = folder
    self._with_jax = with_jax
    self._vistracer = VizTracer(
      output_file=osp.join(folder, 'viztracer', name + '.html'),
      max_stack_depth=10
    )
    self._jax_folder = osp.join(folder, 'jax_profiler/' + name)

  def start(self) -> None:
    """Start to trace."""
    if self._with_jax:
      jax.profiler.start_trace(self._jax_folder)
    self._vistracer.start()

  def stop(self) -> None:
    """Stop tracing."""
    self._vistracer.stop()
    if self._with_jax:
      jax.profiler.stop_trace()

  def save(self) -> None:
    """Save the results."""
    self._vistracer.save()

  def stop_and_save(self) -> None:
    """Combine stop and save."""
    self.stop()
    self.save()
