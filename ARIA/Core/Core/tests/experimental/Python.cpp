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

  // Execute.
  try {
    py::exec("c = add(a, b)\n"
             "\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }

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
  Parent parent0;
  Child child0;
  Parent parent1;
  Child child1;
  std::shared_ptr<GrandParent> parent2 = std::make_shared<Parent>();
  std::unique_ptr<GrandParent> child2 = std::make_unique<Child>();

  locals["parent0"] = parent0; // Pass by copy.
  locals["child0"] = child0;
  locals["parent1"] = py::cast(parent1, py::return_value_policy::reference); // Pass by reference.
  locals["child1"] = py::cast(child1, py::return_value_policy::reference);
  locals["parent2"] = parent2.get(); // Pass by pointer.
  locals["child2"] = child2.get();

  // Execute.
  try {
    py::exec("assert parent0.value() == 1\n"
             "assert child0.value() == 2\n"
             "assert parent1.value() == 1\n"
             "assert child1.value() == 2\n"
             "assert parent2.value() == 1\n"
             "assert child2.value() == 2\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Constness) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<GrandParent>(main, "GrandParent").def("value", &GrandParent::value);
  py::class_<Parent, GrandParent>(main, "Parent").def("value", &Parent::value);
  py::class_<Child, Parent>(main, "Child").def("value", &Child::value);

  // Define variables.
  const Parent parent1;
  const Child child1;
  std::shared_ptr<const GrandParent> parent2 = std::make_shared<const Parent>();
  std::unique_ptr<const GrandParent> child2 = std::make_unique<const Child>();

  locals["parent1"] = py::cast(parent1, py::return_value_policy::reference); // Pass by reference.
  locals["child1"] = py::cast(child1, py::return_value_policy::reference);
  locals["parent2"] = parent2.get(); // Pass by pointer.
  locals["child2"] = child2.get();

  static_assert(std::is_same_v<decltype(parent2.get()), const GrandParent *>);

  // Execute.
  // TODO: Like sol2 for Lua, pybind11 bypasses the `const` requirement.
  try {
    py::exec("assert parent1.value() == 1\n"
             "assert child1.value() == 2\n"
             "assert parent2.value() == 1\n"
             "assert child2.value() == 2\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

} // namespace ARIA
