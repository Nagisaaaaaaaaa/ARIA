#include "ARIA/Property.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace ARIA {

namespace {} // namespace

TEST(Python, Base) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<int>>(main, "vector");

  // Define functions.
  locals["add"] = py::cpp_function([](const std::vector<int> &a, std::vector<int> &b) {
    size_t size = a.size();
    ARIA_ASSERT(size == b.size());

    std::vector<int> c(size);
    for (size_t i = 0; i < size; ++i)
      c[i] = a[i] + b[i];

    return c;
  });

  // Define variables.
  std::vector<int> a = {1, 2, 3};
  std::vector<int> b = {4, 6, 9};

  locals["a"] = py::cast(a, py::return_value_policy::reference);
  locals["b"] = py::cast(b, py::return_value_policy::reference);

  //
  //
  //
  // Execute.
  py::exec("c = add(a, b)\n"
           "\n",
           py::globals(), locals);

  auto c = locals["c"].cast<std::vector<int>>();
  for (const auto &v : c) {
    std::cout << v << std::endl;
  }
}

} // namespace ARIA
