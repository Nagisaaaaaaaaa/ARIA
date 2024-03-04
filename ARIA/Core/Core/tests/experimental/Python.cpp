#include "ARIA/Property.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace ARIA {

namespace {

class GrandParent {
public:
  virtual ~GrandParent() = default;

  virtual int value() = 0;
};

class Parent : public GrandParent {
public:
  virtual ~Parent() = default;

  int value() override { return 1; }
};

class Child final : public Parent {
public:
  virtual ~Child() = default;

  int value() final { return 2; }
};

} // namespace

//
//
//
//
//
TEST(Python, Base) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<int>>(main, "std::vector<int>");

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
  EXPECT_EQ(c[0], 5);
  EXPECT_EQ(c[1], 8);
  EXPECT_EQ(c[2], 12);
}

TEST(Python, Inheritance) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<GrandParent>(main, "GrandParent").def("value", &GrandParent::value);
  py::class_<Parent, GrandParent>(main, "Parent").def("value", &Parent::value);
  py::class_<Child, Parent>(main, "Child").def("value", &Child::value);

  // Define variables.
  std::shared_ptr<GrandParent> parent = std::make_shared<Parent>();
  std::unique_ptr<GrandParent> child = std::make_unique<Child>();

  locals["parent"] = parent.get();
  locals["child"] = child.get();

  //
  //
  //
  // Execute.
  py::exec("assert parent.value() == 1\n"
           "assert child.value() == 2\n",
           py::globals(), locals);
}

} // namespace ARIA
