#include "ARIA/Property.h"

#include <gtest/gtest.h>
#include <nanobind/eval.h>

namespace nb = nanobind;

namespace ARIA {

namespace {} // namespace

TEST(Python, Base) {
  Py_Initialize();
  nb::module_ module = nb::module_::import_("__main__");
  nb::object scope = module.attr("__dict__");

  module.def("add", [](int a, int b) { return a + b; }, nb::arg("a"), nb::arg("b"));

  nb::exec("a = add(1, 2)\n", scope);
}

} // namespace ARIA
