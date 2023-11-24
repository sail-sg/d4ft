#include "eri_kernel.h"

#include "d4ft/native/xla/custom_call.h"


PYBIND11_MODULE(eri_kernel, m) {
  // py::class_<Parent>(m, "Parent").def(py::init<>());
  REGISTER_XLA_FUNCTION(m, Hartree_32);
  REGISTER_XLA_FUNCTION(m, Hartree_64);
  // REGISTER_XLA_MEMBER(m, Parent, ExampleMember);
}
