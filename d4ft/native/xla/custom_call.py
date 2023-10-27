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

import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client


def shape_with_layout(
  specs: List[Tuple[Any, List[int]]]
) -> Tuple[xla_client.Shape, ...]:
  """takes a list of tuples (specs), where each tuple contains a data type
  (dtype) and a shape (a list of integers), and returns a tuple of shapes
  in XLA's format.
  """
  return tuple(
    xla_client.Shape.array_shape(
      dtype,
      shape,
      tuple(range(len(shape) - 1, -1, -1)),  # layout
    ) if len(shape) > 0 else xla_client.Shape.scalar_shape(dtype)
    for dtype, shape in specs
  )


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

    # Register the XLA custom calls
    xla_client.register_custom_call_target(
      f"{name}_cpu".encode(),
      fn=cpu_capsule,
      platform="cpu",
    )
    xla_client.register_custom_call_target(
      f"{name}_gpu".encode(),
      fn=gpu_capsule,
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

      if not hasattr(self, "_state"):
        input_shapes = tuple(arg.shape for arg in args)
        output_shapes = self._shape_inference(input_shapes)
        self._state = self._opaque(input_shapes, output_shapes)

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

    def translation(
      self: Any, c: Any, *args: List[jax.Array], platform: str = "cpu"
    ) -> Any:
      """Defines how the function will be translated into an XLA computation on
      a specific platform (CPU or GPU)."""
      opaque = args[0]
      opaque_spec = c.get_shape(opaque)
      input_specs = tuple(c.get_shape(a) for a in args)
      input_shapes = tuple(spec.dimensions() for spec in input_specs)
      input_dtypes = tuple(spec.element_type() for spec in input_specs)
      output_dtypes = self._output_dtypes()
      output_shapes = self._shape_inference(input_shapes[1:])
      if base._is_member:
        output_dtypes = (opaque_spec.element_type(),) + output_dtypes
        output_shapes = (opaque_spec.dimensions(),) + output_shapes
      input_specs = list(zip(input_dtypes, input_shapes))
      output_specs = list(zip(output_dtypes, output_shapes))

      output_shape_with_layout = shape_with_layout(output_specs)
      if len(output_shape_with_layout) == 1:
        output_shape = output_shape_with_layout[0]
      else:
        output_shape = xla_client.Shape.tuple_shape(output_shape_with_layout)
      return xla_client.ops.CustomCallWithLayout(
        c,
        f"{name}_{platform}".encode(),
        operands=args,
        operand_shapes_with_layout=shape_with_layout(input_specs),
        shape_with_layout=output_shape,
        opaque=self._state.tobytes(),
        has_side_effect=base._is_member,
      )

    attrs["__call__"] = call
    attrs["abstract"] = abstract
    attrs["translation"] = translation
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
      self.prim = core.Primitive(name)
      self.prim.multiple_results = (num_outputs > 1)
      self.prim.def_impl(partial(xla.apply_primitive, self.prim))

      self.prim.def_abstract_eval(self.abstract)

      # Connect the XLA translation rules for JIT compilation
      xla.backend_specific_translations["cpu"][self.prim] = partial(
        self.translation, platform="cpu"
      )
      xla.backend_specific_translations["gpu"][self.prim] = partial(
        self.translation, platform="gpu"
      )

    setattr(subcls, "__init__", init)  # noqa: B010
    return subcls
