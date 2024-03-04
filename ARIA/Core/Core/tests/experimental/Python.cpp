#include "ARIA/Property.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/operators.h>

namespace py = pybind11;

namespace ARIA {

namespace {

struct GrandParent {
  virtual ~GrandParent() = default;

  virtual int value() = 0;
};

struct Parent : public GrandParent {
  virtual ~Parent() = default;

  int value() override { return 1; }
};

struct Child final : public Parent {
  virtual ~Child() = default;

  int value() final { return 2; }
};

struct OverloadWithConst {
  int value() const { return 0; }

  int value() { return 1; }
};

struct OverloadWithParameters {
  int value(int) { return 0; }

  std::string value(double) { return "0"; }

  std::vector<std::string> value(int, double) { return {"0"}; }
};

class Object {
public:
  std::vector<std::string> &name0() { return name_; }

  std::vector<std::string> &get_name0() { return name_; }

  void set_name0(const std::vector<std::string> &value) { name_ = value; }

public:
  ARIA_PROP(public, public, , std::vector<std::string>, name1);

private:
  std::vector<std::string> name_ = {"Python です喵"}; // Test UTF-8.

  std::vector<std::string> ARIA_PROP_IMPL(name1)() const { return name_; }

  std::vector<std::string> ARIA_PROP_IMPL(name1)(const std::vector<std::string> &name) { name_ = name; }
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

TEST(Python, Const) {
  // TODO: Pybind11 always bypasses the `const` requirement.
  // TODO: For overloaded methods where `const` is the only difference,
  //       the earlier-defined one will be selected.

  // Bypass const.
  {
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
    static_assert(std::is_same_v<decltype(child2.get()), const GrandParent *>);

    // Execute.
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

  // Give const version higher priority.
  {
    py::scoped_interpreter guard{};

    // Get scope.
    py::object main = py::module_::import("__main__");
    py::dict locals;

    // Define types.
    py::class_<OverloadWithConst>(main, "OverloadWithConst")
        .def("value",
             static_cast<decltype(std::declval<const OverloadWithConst>().value()) (OverloadWithConst::*)() const>(
                 &OverloadWithConst::value))
        .def("value", static_cast<decltype(std::declval<OverloadWithConst>().value()) (OverloadWithConst::*)()>(
                          &OverloadWithConst::value));

    // Define variables.
    const OverloadWithConst overloadConst;
    OverloadWithConst overloadNonConst;

    locals["overloadConst"] = py::cast(overloadConst, py::return_value_policy::reference); // Pass by reference.
    locals["overloadNonConst"] = py::cast(overloadNonConst, py::return_value_policy::reference);

    // Execute.
    try {
      py::exec("assert overloadConst.value() == 0\n"
               "assert overloadNonConst.value() == 0\n",
               py::globals(), locals);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }

  // Give non-const version higher priority.
  {
    py::scoped_interpreter guard{};

    // Get scope.
    py::object main = py::module_::import("__main__");
    py::dict locals;

    // Define types.
    py::class_<OverloadWithConst>(main, "OverloadWithConst")
        .def("value", static_cast<decltype(std::declval<OverloadWithConst>().value()) (OverloadWithConst::*)()>(
                          &OverloadWithConst::value))
        .def("value",
             static_cast<decltype(std::declval<const OverloadWithConst>().value()) (OverloadWithConst::*)() const>(
                 &OverloadWithConst::value));

    // Define variables.
    const OverloadWithConst overloadConst;
    OverloadWithConst overloadNonConst;

    locals["overloadConst"] = py::cast(overloadConst, py::return_value_policy::reference); // Pass by reference.
    locals["overloadNonConst"] = py::cast(overloadNonConst, py::return_value_policy::reference);

    // Execute.
    try {
      py::exec("assert overloadConst.value() == 1\n"
               "assert overloadNonConst.value() == 1\n",
               py::globals(), locals);
    } catch (std::exception &e) {
      fmt::print("{}\n", e.what());
      EXPECT_FALSE(true);
    }
  }
}

TEST(Python, Overload) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<std::string>>(main, "std::vector<std::string>").def(py::self == py::self);

  py::class_<OverloadWithParameters>(main, "OverloadWithParameters")
      .def("value", static_cast<decltype(std::declval<OverloadWithParameters>().value(std::declval<int>())) (
                        OverloadWithParameters::*)(int)>(&OverloadWithParameters::value))
      .def("value", static_cast<decltype(std::declval<OverloadWithParameters>().value(std::declval<double>())) (
                        OverloadWithParameters::*)(double)>(&OverloadWithParameters::value))
      .def("value", static_cast<decltype(std::declval<OverloadWithParameters>().value(
                        std::declval<int>(), std::declval<double>())) (OverloadWithParameters::*)(int, double)>(
                        &OverloadWithParameters::value));

  // Define variables.
  OverloadWithParameters overload;

  locals["overload"] = py::cast(overload, py::return_value_policy::reference);
  locals["vector"] = std::vector<std::string>{"0"};

  EXPECT_TRUE(overload.value(0, 0.0) == std::vector<std::string>{"0"});

  // Execute.
  try {
    py::exec("assert overload.value(0) == 0\n"
             "assert overload.value(0.0) == '0'\n"
             "assert overload.value(0, 0.0) == vector\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

TEST(Python, Properties) {
  py::scoped_interpreter guard{};

  // Get scope.
  py::object main = py::module_::import("__main__");
  py::dict locals;

  // Define types.
  py::class_<std::vector<std::string>>(main, "std::vector<std::string>")
      .def(py::self == py::self)
      .def("clear", &std::vector<std::string>::clear);

  py::class_<Object>(main, "Object")
      .def("name0", static_cast<decltype(std::declval<Object>().name0()) (Object::*)()>(&Object::name0))
      .def("name1", static_cast<decltype(std::declval<Object>().name1()) (Object::*)()>(&Object::name1))
      .def_property("name0_prop", &Object::get_name0, &Object::set_name0);

  py::class_<decltype(std::declval<Object>().name1())>(main, "Object::name1")
      .def("value",
           static_cast<decltype(std::declval<decltype(std::declval<Object>().name1())>().value()) (
               decltype(std::declval<Object>().name1())::*)() const>(&decltype(std::declval<Object>().name1())::value))
      .def(py::self == py::self)
      .def(py::self == std::vector<std::string>())
      .def(std::vector<std::string>() == py::self);

  // Define variables.
  std::vector<std::string> nameCase0 = {"Python です喵"};
  std::vector<std::string> nameCase1 = {"Python 喵です"};
  std::vector<std::string> nameCase2 = {};
  Object obj;

  locals["nameCase0"] = py::cast(nameCase0, py::return_value_policy::reference);
  locals["nameCase1"] = py::cast(nameCase1, py::return_value_policy::reference);
  locals["nameCase2"] = py::cast(nameCase2, py::return_value_policy::reference);
  locals["obj"] = py::cast(obj, py::return_value_policy::reference);

  // Execute.
  try {
    py::exec("assert obj.name0() == nameCase0\n" // Test getter.
             "assert obj.name1() == nameCase0\n"
             "assert obj.name0_prop == nameCase0\n"
             "assert nameCase0 == obj.name0()\n"
             "assert nameCase0 == obj.name1()\n"
             "assert nameCase0 == obj.name0_prop\n"
             "\n"
             "obj.name0_prop = nameCase1\n" // Test setter with `operator=`.
             "assert obj.name0() == nameCase1\n"
             "assert obj.name1() == nameCase1\n"
             "assert obj.name0_prop == nameCase1\n"
             "assert nameCase1 == obj.name0()\n"
             "assert nameCase1 == obj.name1()\n"
             "assert nameCase1 == obj.name0_prop\n"
             "\n"
             "obj.name0_prop.clear()\n" // Test setter with `clear()`.
             "assert obj.name0() == nameCase2\n"
             "assert obj.name1() == nameCase2\n"
             "assert obj.name0_prop == nameCase2\n"
             "assert nameCase2 == obj.name0()\n"
             "assert nameCase2 == obj.name1()\n"
             "assert nameCase2 == obj.name0_prop\n",
             py::globals(), locals);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

} // namespace ARIA
