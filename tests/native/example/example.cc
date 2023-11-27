#include "example.h"

#include "d4ft/native/xla/custom_call.h"

PYBIND11_MODULE(example, m) {
  py::class_<Parent>(m, "Parent").def(py::init<>());
  REGISTER_XLA_FUNCTION(m, Example);
  REGISTER_XLA_MEMBER(m, Parent, ExampleMember);
}
