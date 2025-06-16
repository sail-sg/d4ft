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

from abc import ABCMeta
from functools import partial
from typing import Any, Dict, List, Tuple
import warnings

import jax
from jax import dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir
from jax.lib import xla_client

# Handle JAX 0.6.0+ API changes
try:
  from jax import core
  Primitive = core.Primitive
except AttributeError:
  # JAX 0.6.0+ moved Primitive to jax.extend.core
  from jax import extend
  from jax import core
  Primitive = extend.core.Primitive


def aval_to_mlir_type(aval):
  """Convert JAX abstract value to MLIR type."""
  return mlir.aval_to_ir_type(aval)


class CustomCallMeta(ABCMeta):
  """Meta class that creates classes that represent a XLA custom call
  from the pybind module that implements the CPU and GPU capsules.
  """

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    """Creates a new class that represents a XLA custom call.

    Args:
      cls: self.
      name: name of the custom op.
      parents: list of parent class. Must be a singleton that contains the
        pybind module that implements the CPU and GPU capsules.
      attrs: attributes of the class.
    """
    # get capsules from the pybind module
    assert len(parents) == 1
    base = parents[0]
    cpu_capsule, gpu_capsule = base._capsules

    # Register the custom call targets
    # JAX 0.6.0 still has register_custom_call_target but it's in jaxlib
    try:
      # Try the old location first
      register_fn = xla_client.register_custom_call_target
    except AttributeError:
      # In newer versions, it might be in a different location
      try:
        from jaxlib import xla_client as jaxlib_xla_client
        register_fn = jaxlib_xla_client.register_custom_call_target
      except (ImportError, AttributeError):
        register_fn = None
    
    if register_fn:
      register_fn(
        f"{name}_cpu".encode(),
        cpu_capsule,
        platform="cpu",
      )
      register_fn(
        f"{name}_gpu".encode(),
        gpu_capsule,
        platform="gpu",
      )

    def call(self: Any, *args: List[jax.Array]) -> List[jax.Array]:
      """Binded to __call__, which exposes the primitive to user code.

      Checks the data types of input arguments, performs shape inference,
      and binds a primitive operation to the state and arguments"""
      input_dtypes = list(
        map(dtypes.canonicalize_dtype, (arg.dtype for arg in args))
      )
      required_input_dtypes = list(
        map(dtypes.canonicalize_dtype, self._input_dtypes())
      )
      if input_dtypes != required_input_dtypes:
        raise RuntimeError(
          f"Requested {required_input_dtypes}, got {input_dtypes}"
        )

      # if not hasattr(self, "_state"):
      input_shapes = tuple(arg.shape for arg in args)
      output_shapes = self._shape_inference(input_shapes)
      self._state = self._opaque(input_shapes, output_shapes)
      # _state = self._opaque(input_shapes, output_shapes)
      output = self.prim.bind(self._state, *args)
      if base._is_member:
        assert len(output) >= 2
        self._state = output[0]  # update the state
        output = output[1] if len(output) == 2 else output[1:]
      return output

    def abstract(self: Any, *args: List[jax.Array]) -> ShapedArray:
      """Abstract evaluation of the function, which generates the output shapes
      and data types based on the input shapes and data types.
      This enables JIT compilation of the function."""
      output_dtypes = self._output_dtypes()
      output_shapes = self._shape_inference(tuple(a.shape for a in args[1:]))
      if base._is_member:
        output_dtypes = (self._state.dtype,) + output_dtypes
        output_shapes = (self._state.shape,) + output_shapes
      ret = tuple(
        ShapedArray(shape, dtype)
        for dtype, shape in zip(output_dtypes, output_shapes)
      )
      if len(ret) == 1:
        ret = ret[0]
      return ret

    def lowering_rule(
      self: Any, ctx: mlir.LoweringRuleContext, *args, platform: str = "cpu"
    ) -> Any:
      """Defines how the function will be lowered to MLIR custom call."""
      output_dtypes = self._output_dtypes()
      input_shapes = tuple(aval.shape for aval in ctx.avals_in[1:])
      output_shapes = self._shape_inference(input_shapes)
      
      if base._is_member:
        output_dtypes = (ctx.avals_in[0].dtype,) + output_dtypes
        output_shapes = (ctx.avals_in[0].shape,) + output_shapes
      
      # Create output types for MLIR
      output_avals = tuple(
        ShapedArray(shape, dtype) 
        for dtype, shape in zip(output_dtypes, output_shapes)
      )
      result_types = [aval_to_mlir_type(aval) for aval in output_avals]
      
      # For the backend config, we don't serialize the state since it's passed as an operand
      # The backend config should contain static configuration data only
      backend_config = b""
      
      # Create the custom call operation
      # Use api_version=2 for the legacy custom call API
      # Suppress deprecation warning for mlir.custom_call
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        op = mlir.custom_call(
          f"{name}_{platform}",
          operands=args,
          result_types=result_types,
          backend_config=backend_config,
          has_side_effect=base._is_member,
          api_version=2,  # Use legacy API version for custom calls
        )
      return op.results

    attrs["__call__"] = call
    attrs["abstract"] = abstract
    attrs["lowering_rule"] = lowering_rule
    attrs["_cpu_capsule"] = cpu_capsule
    attrs["_gpu_capsule"] = gpu_capsule
    subcls = super().__new__(cls, name, parents, attrs)

    def init(self: Any, parent: Any = None) -> None:
      """Binded to __init__. Create lax prim."""
      if base._is_member:
        super(subcls, self).__init__(parent)
        num_outputs = len(self._output_dtypes()) + 1
      else:
        super(subcls, self).__init__()
        num_outputs = len(self._output_dtypes())

      # create the primitive
      self.prim = Primitive(name)
      self.prim.multiple_results = (num_outputs > 1)
      
      # Use a simple implementation function for the primitive
      def _impl(*args):
        raise NotImplementedError(f"Implementation for {name} not available in eager mode. Use jax.jit().")
      
      self.prim.def_impl(_impl)
      self.prim.def_abstract_eval(self.abstract)

      # Register MLIR lowering rules for JIT compilation  
      mlir.register_lowering(
        self.prim, partial(self.lowering_rule, platform="cpu"), platform="cpu"
      )
      mlir.register_lowering(
        self.prim, partial(self.lowering_rule, platform="gpu"), platform="gpu"
      )

    setattr(subcls, "__init__", init)  # noqa: B010
    return subcls
