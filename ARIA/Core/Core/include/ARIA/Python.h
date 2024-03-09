#pragma once

/// \file
/// \warning `Python` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/detail/PythonImpl.h"

namespace ARIA {

/// \brief While namespace `Python` contains all the wrapped APIs,
/// namespace `py` contains all the raw APIs, for example, `py::exec`.
///
/// Wrapped and raw APIs can be used together safely in ARIA
/// (just like CUDA runtime API and CUDA API wrapper), but
/// we recommend using the wrapped APIs as much as you can.
namespace py = python::detail::py;

//
//
//
/// \brief A RAII class for the Python interpreter.
/// This class is a wrapper for `py::scoped_interpreter`.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
///
/// py::exec("print('Hello Python!')\n");
/// ```
///
/// \details This class should be used instead of `py::scoped_interpreter` because
/// it contains more contexts about Python modules, for example,
/// which C++ types have currently been defined for each module.
/// These contexts are helpful for automatically and recursively defining types.
namespace Python {
using python::detail::ScopedInterpreter;
}

//
//
//
/// \brief A Python module.
/// This class is a wrapper for `py::module`.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
///
/// static_assert(main.HasType<int>());
/// EXPECT_FALSE(main.HasType<std::vector<int>>());
/// ```
///
/// \details This class should be used instead of `py::module_` because
/// it contains more contexts, for example,
/// which C++ types have currently been defined for this module.
/// These contexts are helpful for automatically and recursively defining types.
namespace Python {
using python::detail::Module;
}

//
//
//
/// \brief A Python dictionary, which can be used to
/// represent global or local variables.
///
/// \example ```cpp
/// Python::ScopedInterpreter interpreter{};
/// Python::Module main = interpreter.Import("__main__");
/// Python::Dict local{main};
///
/// // Define a function which adds 2 `std::vector<int>` and returns the sum.
/// local["add"] = py::cpp_function([](const std::vector<int> &a, const std::vector<int> &b) {
///   size_t size = a.size();
///   ARIA_ASSERT(size == b.size());
///
///   std::vector<int> c(size);
///   for (size_t i = 0; i < size; ++i)
///     c[i] = a[i] + b[i];
///
///   return c;
/// });
///
/// // Define 2 variables.
/// std::vector<int> a = {1, 2, 3};
/// std::vector<int> b = {4, 6, 9};
///
/// local["a"] = a;  // `a` is passed to Python by value.
/// local["b"] = &b; // `b` is passed to Python by reference,
///
/// // Call the Python interpreter and computes a + b.
/// py::exec("c = add(a, b)\n",
///          py::globals(), // Global variables: set to default.
///          local);        // Local variables: set to `local`.
///
/// // Get the computation result from Python.
/// auto c = local["c"].Cast<std::vector<int>>();
/// EXPECT_EQ(c[0], 5);
/// EXPECT_EQ(c[1], 8);
/// EXPECT_EQ(c[2], 12);
/// ```
///
/// \details If you are familiar with libraries such as pybind, you may have
/// noticed that `std::vector<int>` is not explicitly defined in this example.
/// That is the magic provided by ARIA `ScopedInterpreter`, `Module`, and `Dict`.
/// Types are automatically and recursively defined at `= a` and `= &b`.
namespace Python {
using python::detail::Dict;
}

//
//
//
//
//
/// \brief The wrapped APIs allow automatically and recursively defining Python types,
/// as introduced in the comments of `Dict`.
/// To make it work, we should define how a C++ class looks like in Python,
/// for example, which methods and properties should be defined.
/// This is done within the region of `ARIA_PYTHON_TYPE_BEGIN` and `ARIA_PYTHON_TYPE_END`.
///
/// \example ```cpp
/// class Object { ... };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ...
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_BEGIN(type) __ARIA_PYTHON_TYPE_BEGIN(type)

//
//
//
/// \brief A template version of `ARIA_PYTHON_TYPE_BEGIN`.
///
/// \example ```cpp
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(std::vector);
/// ...
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_) __ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_)

//
//
//
/// \brief We usually want to define private methods to Python,
/// especially while defining ARIA properties, whose
/// getters and setters are always private.
///
/// This macro should be used to give the Python APIs
/// accessibility to private methods.
///
/// \example ```
/// class Object {
/// private:
///   ...
///   ARIA_PYTHON_TYPE_FRIEND;
/// };
/// ```
#define ARIA_PYTHON_TYPE_FRIEND __ARIA_PYTHON_TYPE_FRIEND

//
//
//
/// \brief Define a method for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   const std::string& name() const { ... }
/// };
///
/// template <typename T, auto size>
/// class Vec {
/// public:
///   void Normalize() { ... }
///   Vec<T, size> Dot(const Vec<T, size>& others) const { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_METHOD(const, name);
/// ARIA_PYTHON_TYPE_END;
///
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_METHOD(, Normalize);
/// // Here, `T` is a place holder equals to the instantiated template type, that is, `Vec<T, size>`.
/// ARIA_PYTHON_TYPE_METHOD(const, Dot, T);
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_METHOD /* (specifiers, name, parameters...) */ __ARIA_PYTHON_TYPE_METHOD

//
//
//
/// \brief Define an ARIA property for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   ARIA_PROP(public, public, , std::string, name);
/// private:
///   std::string ARIA_PROP_IMPL(name)() const { ... }
///   void ARIA_PROP_IMPL(name)(const std::string &name) { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_PROPERTY(name);
/// ARIA_PYTHON_TYPE_END;
#define ARIA_PYTHON_TYPE_PROPERTY(name) __ARIA_PYTHON_TYPE_PROPERTY(name)

//
//
//
/// \brief Define a readonly ARIA property for the given type or template.
///
/// \example ```cpp
/// class Object {
/// public:
///   ARIA_PROP(public, private, , std::string, name);
/// private:
///   std::string ARIA_PROP_IMPL(name)() const { ... }
/// };
///
/// ARIA_PYTHON_TYPE_BEGIN(Object);
/// ARIA_PYTHON_TYPE_READONLY_PROPERTY(name);
/// ARIA_PYTHON_TYPE_END;
#define ARIA_PYTHON_TYPE_READONLY_PROPERTY(name) __ARIA_PYTHON_TYPE_READONLY_PROPERTY(name)

//
//
//
/// \brief Define an unary operator for the given type or template.
///
/// \example ```cpp
/// ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(Vec);
/// ARIA_PYTHON_TYPE_UNARY_OPERATOR(+); // Enables `b = +a` in Python.
/// ARIA_PYTHON_TYPE_UNARY_OPERATOR(-); // Enables `b = -a` in Python.
/// ARIA_PYTHON_TYPE_END;
/// ```
#define ARIA_PYTHON_TYPE_UNARY_OPERATOR(op) __ARIA_PYTHON_TYPE_UNARY_OPERATOR(op)

//
//
//
#define ARIA_PYTHON_TYPE_BINARY_OPERATOR /* (op) or (op, others) */ __ARIA_PYTHON_TYPE_BINARY_OPERATOR

//
//
//
/// \brief Finish the definition of how a C++ class looks like in Python.
/// This macro should be used together with `ARIA_PYTHON_TYPE_BEGIN`.
///
/// \see ARIA_PYTHON_TYPE_BEGIN
#define ARIA_PYTHON_TYPE_END __ARIA_PYTHON_TYPE_END

//
//
//
#define ARIA_PYTHON_ADD_TYPE /* (type) or (type, module) */ __ARIA_PYTHON_ADD_TYPE

} // namespace ARIA
