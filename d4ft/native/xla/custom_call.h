/*
 * Copyright 2023 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime_api.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "specs.h"

namespace py = pybind11;

/**
 * @brief Introspection (traits) of a tuple of Specs, i.e. getting the shapes
 * and dtype, and converts it to pybind Specs.
 * */
template <typename SpecsTuple>
struct SpecsTraits;

template <typename... Specs>
struct SpecsTraits<std::tuple<Specs...>> {
  using Shapes = std::tuple<decltype(Specs::shape)...>;
  using Dtypes = std::tuple<typename Specs::dtype...>;

  /**
   * @brief Returns a tuple of py::dtypes
   * */
  static auto ToDtypes() {
    return std::make_tuple(py::dtype::of<typename Specs::dtype>()...);
  }

  /**
   * @brief Builds a tuple of Spec from shapes
   * */
  static auto FromShapes(const std::tuple<decltype(Specs::shape)...>& shapes) {
    return std::apply(
        [](auto&&... shape) { return std::make_tuple(Specs(shape)...); },
        shapes);
  }
};

/**
 * @brief Introspection (traits) of a function pointer, i.e. getting the
 * input/output types and class.
 * */
template <typename F>
struct FunctionTraits;

/**
 * @brief function pointers
 * */
template <typename Ret, typename... Args>
struct FunctionTraits<Ret (*)(const Args&...)> {
  using Class = void;
  using ArgsTuple = std::tuple<Args...>;
  using Return = Ret;
};

/**
 * @brief member function pointers
 * */
template <typename Cls, typename Ret, typename... Args>
struct FunctionTraits<Ret (Cls::*)(const Args&...)> {
  using Class = Cls;
  using ArgsTuple = std::tuple<Args...>;
  using Return = Ret;
};

/**
 * @brief Creates an array of Array / Tensor from an array of buffer pointers,
 * and a tuple of specifications.
 *
 * @tparam T type of the element in the buffers
 * @tparam Specs a parameter pack representing types of the specifications for
 * the buffers
 * @tparam Is an index sequence.
 *
 * @param buffers a pointer to an array of buffer pointers
 * @param specs a pointer to a tuple of specifications
 * */
template <typename T, typename... Specs, std::size_t... Is>
decltype(auto) PrepareArray(T** buffers, std::tuple<Specs...>* specs,
                            std::index_sequence<Is...>) {
  if constexpr (std::is_const_v<T>) {  // constant type
    return std::apply(
        [&](auto&&... spec) {
          return std::make_tuple(Array<const typename Specs::dtype>{
              reinterpret_cast<const typename Specs::dtype*>(buffers[Is]),
              &spec}...);
        },
        *specs);
  } else {  // not constant type
    return std::apply(
        [&](auto&&... spec) {
          return std::make_tuple(Array<typename Specs::dtype>{
              reinterpret_cast<typename Specs::dtype*>(buffers[Is]), &spec}...);
        },
        *specs);
  }
}

/**
 * @brief use sizeof...(Specs) to get the number of buffers, and call the above
 * template
 * */
template <typename T, typename... Specs>
decltype(auto) PrepareArray(T** buffers, std::tuple<Specs...>* specs) {
  return PrepareArray(buffers, specs,
                      std::make_index_sequence<sizeof...(Specs)>{});
}

/**
 * @brief Provides CustomCall with methods to encode/decode the input/output
 * shape into an opaque object.
 * */
template <typename CustomCall, typename Parent = void>
class Xla : public CustomCall {
 public:
  // introspect the ShapeInference function of the CustomCall
  using Traits = FunctionTraits<decltype(&CustomCall::ShapeInference)>;
  using InputSpecs = typename Traits::ArgsTuple;
  using OutputSpecs = typename Traits::Return;

  // introspect the input/output shape of ShapeInference
  using InputShapes = typename SpecsTraits<InputSpecs>::Shapes;
  using OutputShapes = typename SpecsTraits<OutputSpecs>::Shapes;

  // type deduction of the input/output arrays
  using Inputs = decltype(PrepareArray(std::declval<const void**>(),
                                       std::declval<InputSpecs*>()));
  using Outputs = decltype(PrepareArray(std::declval<void**>(),
                                        std::declval<OutputSpecs*>()));

  // whether the CustomCall is a member function
  static constexpr bool IsMember = !std::is_same<Parent, void>::value;

  // number of inputs and outputs
  static constexpr auto NumInputs = std::tuple_size_v<InputSpecs>;
  static constexpr auto NumOutputs = std::tuple_size_v<OutputSpecs>;

  // if the CustomCall is not a member function, it must has a return value
  static_assert(IsMember || NumOutputs > 0, "CustomCall has no return value.");

  /**
   * constructors for non-member functions
   * */
  template <bool IM = IsMember, std::enable_if_t<!IM, bool> = true>
  explicit Xla() {}

  /**
   * constructors for member functions
   * */
  template <bool IM = IsMember, std::enable_if_t<IM, bool> = true>
  explicit Xla(Parent* p) : CustomCall(p) {}

  /**
   * py::dtypes of the input arguments.
   */
  auto InputDtypes() {
    return SpecsTraits<typename Traits::ArgsTuple>::ToDtypes();
  }

  /**
   * py::dtypes of the outputs.
   */
  auto OutputDtypes() {
    return SpecsTraits<typename Traits::Return>::ToDtypes();
  }

  /**
   * Infer the output shapes given input shapes.
   */
  auto ShapeInference(const InputShapes& shapes) {
    OutputSpecs output_specs;
    if constexpr (IsMember) {  // member function
      auto cc = static_cast<CustomCall*>(this);
      output_specs = std::apply(
          [&](auto&&... args) { return cc->ShapeInference(args...); },
          SpecsTraits<InputSpecs>::FromShapes(shapes));
    } else {  // non-member function, can use the static function directly
      output_specs = std::apply(CustomCall::ShapeInference,
                                SpecsTraits<InputSpecs>::FromShapes(shapes));
    }
    return std::apply(
        [](auto&&... spec) { return std::make_tuple(spec.shape...); },
        output_specs);
  }

  /**
   * Encode the input/output shape into an opaque object that is returned as a
   * py::array object to the Python side
   */
  py::array Opaque(const InputShapes& input_shapes,
                   const OutputShapes& output_shapes) {
    std::vector<uint8_t> out;
    CustomCall* obj = static_cast<CustomCall*>(this);
    // encode and store the pointer to the CustomCall object
    auto opaquep = reinterpret_cast<uint8_t*>(&obj);
    out.insert(out.end(), opaquep, opaquep + sizeof(CustomCall*));
    // encode and store the input/output shapes
    auto encode_shape = [&](const Shape& shape) {
      int rank = shape.size();
      auto* rankp = reinterpret_cast<const uint8_t*>(&rank);
      out.insert(out.end(), rankp, rankp + sizeof(int));
      auto* shapep = reinterpret_cast<const uint8_t*>(shape.data());
      out.insert(out.end(), shapep, shapep + rank * sizeof(int));
    };
    std::apply([&](auto&&... shape) { (encode_shape(shape), ...); },
               std::tuple_cat(input_shapes, output_shapes));
    // move the encoding into a py::array object
    auto* ptr = new std::vector<uint8_t>(std::move(out));
    auto capsule = py::capsule(ptr, [](void* ptr) {
      delete reinterpret_cast<std::vector<uint8_t>*>(ptr);
    });
    return py::array(std::vector<std::size_t>({ptr->size()}), ptr->data(),
                     capsule);
  }

  /**
   * Decode the opaque data encoded by the Opaque function above.
   */
  static std::tuple<CustomCall*, InputSpecs, OutputSpecs, int> DecodeOpaque(
      void* opaque) {
    // decode the pointer to the CustomCall object
    int total_bytes = sizeof(CustomCall*);
    auto* data = reinterpret_cast<uint8_t*>(opaque);
    CustomCall* obj = *reinterpret_cast<CustomCall**>(data);
    data += sizeof(CustomCall*);
    // decode the input/output shapes
    auto decode_shape = [&](int** data, Shape* shape) {
      int size = *reinterpret_cast<int*>(*data);
      total_bytes += sizeof(int) * (size + 1);
      *data += 1;
      shape->insert(shape->end(), *data, *data + size);
      *data += size;
    };
    auto* int_ptr = reinterpret_cast<int*>(data);
    InputSpecs input_specs;
    OutputSpecs output_specs;
    std::apply(
        [&](auto&&... spec) { (decode_shape(&int_ptr, &spec.shape), ...); },
        input_specs);
    std::apply(
        [&](auto&&... spec) { (decode_shape(&int_ptr, &spec.shape), ...); },
        output_specs);
    return std::make_tuple(obj, input_specs, output_specs, total_bytes);
  }

  /**
   * @brief CPU version of the custom XLA operation.
   *
   * Although the opaque API is only required for the Gpu backend, we
   * choose to use it for the Cpu backend for uniformity.
   *
   * The opaque data is stored in the first input buffer (in[0]), which
   * contains a CustomCall object pointer, InputSpecs, OutputSpecs, and the
   * length of the opaque data.
   *
   * @param out pointer to the output buffer
   * @param in pointer to the array of input buffer, which are the encoded
   * opaque data from python side
   */
  static void Cpu(void* out, const void** in) {
    // Parse opaque data from in[0]
    CustomCall* obj;
    InputSpecs input_specs;
    OutputSpecs output_specs;
    int opaque_len;
    std::tie(obj, input_specs, output_specs, opaque_len) =
        DecodeOpaque(const_cast<void*>(in[0]));
    in += 1;
    // Load in[1:] into tuple of Array
    auto inputs = PrepareArray(in, &input_specs);
    Outputs outputs;
    if constexpr (IsMember) {           // casting member function into XLA
      if constexpr (NumOutputs == 0) {  // no outputs, just copy the pointer
        std::memcpy(out, &obj, opaque_len);
      } else {  // has outputs, prepare the output buffer
        void** outs = reinterpret_cast<void**>(out);
        std::memcpy(outs[0], &obj, opaque_len);
        outputs = PrepareArray(outs + 1, &output_specs);
      }
      // calls the CPU implementation of the member function
      std::apply([&](auto&&... arg) { obj->Cpu(arg...); },
                 std::tuple_cat(inputs, outputs));
    } else {  // casting static function into XLA
      if constexpr (NumOutputs == 1) {
        outputs = PrepareArray(&out, &output_specs);
      } else {
        void** outs = reinterpret_cast<void**>(out);
        outputs = PrepareArray(outs, &output_specs);
      }
      // calls the CPU implementation of the member function
      std::apply([&](auto&&... arg) { CustomCall::Cpu(arg...); },
                 std::tuple_cat(inputs, outputs));
    }
  }

  /**
   * @brief GPU version of the custom XLA operation.
   *
   * @param stream The CUDA stream on which the kernel should run.
   * @param buffers Pointers to the input and output memory regions.
   * @param opaque A pointer to the opaque data that contains the serialized
   * form of the additional parameters needed for the kernel execution.
   * @param opaque_len The length of the opaque data
   */
  static void Gpu(cudaStream_t stream, void** buffers, const char* opaque,
                  std::size_t opaque_len) {
    // Parse opaque data
    CustomCall* obj;
    InputSpecs input_specs;
    OutputSpecs output_specs;
    int real_opaque_len;  // not used
    std::tie(obj, input_specs, output_specs, real_opaque_len) =
        DecodeOpaque(static_cast<void*>(const_cast<char*>(opaque)));
    // Load buffers[1:1+len(inputs)] as inputs
    buffers += 1;
    auto inputs = PrepareArray(const_cast<const void**>(buffers), &input_specs);
    Outputs outputs;
    buffers += NumInputs;
    if constexpr (IsMember) {
      buffers += 1;
    }
    // Load buffers[1+len(inputs)(+1 for member func):] as outputs
    outputs = PrepareArray(buffers, &output_specs);
    if constexpr (IsMember) {
      std::apply([&](auto&&... arg) { obj->Gpu(stream, arg...); },
                 std::tuple_cat(inputs, outputs));
    } else {
      std::apply([&](auto&&... arg) { CustomCall::Gpu(stream, arg...); },
                 std::tuple_cat(inputs, outputs));
    }
  }

  /**
   * The Xla::Cpu and Xla::Gpu functions are being exposed to Python code
   * through these capsules, and they are being identified by the name
   * "xla._CUSTOM_CALL_TARGET".
   */
  static std::vector<py::capsule> capsules;
};

template <typename CustomCall, typename Parent>
std::vector<py::capsule> Xla<CustomCall, Parent>::capsules =
    std::vector<py::capsule>({py::capsule(reinterpret_cast<void*>(Xla::Cpu),
                                          "xla._CUSTOM_CALL_TARGET"),
                              py::capsule(reinterpret_cast<void*>(Xla::Gpu),
                                          "xla._CUSTOM_CALL_TARGET")});

py::object abc_meta = py::module::import("abc").attr("ABCMeta");

/**
 * This macro registers a class in Python that is a template specialization of
 * Xla with CustomCall as the template argument. The Python class is named
 * _<CustomCall>. This class has a constructor (py::init<>()), and several
 * static and instance methods exposed to Python.
 * */
#define REGISTER_XLA_FUNCTION(MODULE, CustomCall)                    \
  py::class_<Xla<CustomCall>>(MODULE, "_" #CustomCall,               \
                              py::metaclass(abc_meta))               \
      .def(py::init<>())                                             \
      .def_readonly_static("_capsules", &Xla<CustomCall>::capsules)  \
      .def_readonly_static("_is_member", &Xla<CustomCall>::IsMember) \
      .def("_opaque", &Xla<CustomCall>::Opaque)                      \
      .def("_input_dtypes", &Xla<CustomCall>::InputDtypes)           \
      .def("_output_dtypes", &Xla<CustomCall>::OutputDtypes)         \
      .def("_shape_inference", &Xla<CustomCall>::ShapeInference);

/**
 * This macro registers a class that is a template specialization of Xla with
 * CustomCall and Parent as template arguments. This version of Xla is designed
 * to represent member functions of Parent.
 * */
#define REGISTER_XLA_MEMBER(MODULE, Parent, CustomCall)                      \
  py::class_<Xla<CustomCall, Parent>>(MODULE, "_" #CustomCall,               \
                                      py::metaclass(abc_meta))               \
      .def(py::init<Parent*>())                                              \
      .def_readonly_static("_capsules", &Xla<CustomCall, Parent>::capsules)  \
      .def_readonly_static("_is_member", &Xla<CustomCall, Parent>::IsMember) \
      .def("_opaque", &Xla<CustomCall, Parent>::Opaque)                      \
      .def("_input_dtypes", &Xla<CustomCall, Parent>::InputDtypes)           \
      .def("_output_dtypes", &Xla<CustomCall, Parent>::OutputDtypes)         \
      .def("_shape_inference", &Xla<CustomCall, Parent>::ShapeInference);
