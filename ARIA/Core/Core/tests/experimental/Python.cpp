#include "ARIA/Property.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace ARIA {

namespace {

int add(int a, int b) {
  return a + b;
}

PYBIND11_EMBEDDED_MODULE(test, m) {
  m.def("add", [](int i, int j) { return i + j; });
}

} // namespace

TEST(Python, Base) {
  py::scoped_interpreter guard{};

  auto test = py::module_::import("test");

  py::exec("import test\n"
           "\n"
           "a = 5\n"
           "b = 6\n"
           "c = test.add(a, b)\n"
           "print(c)\n");
}

} // namespace ARIA
