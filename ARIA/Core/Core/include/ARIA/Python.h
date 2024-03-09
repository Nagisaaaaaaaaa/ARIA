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
#define ARIA_PYTHON_TYPE_BEGIN(type) __ARIA_PYTHON_TYPE_BEGIN(type)

//
//
//
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
#define ARIA_PYTHON_TYPE_METHOD /* (specifiers, name, parameters...) */ __ARIA_PYTHON_TYPE_METHOD

//
//
//
#define ARIA_PYTHON_TYPE_PROPERTY(name) __ARIA_PYTHON_TYPE_PROPERTY(name)

//
//
//
#define ARIA_PYTHON_TYPE_READONLY_PROPERTY(name) __ARIA_PYTHON_TYPE_READONLY_PROPERTY(name)

//
//
//
#define ARIA_PYTHON_TYPE_UNARY_OPERATOR(op) __ARIA_PYTHON_TYPE_UNARY_OPERATOR(op)

//
//
//
#define ARIA_PYTHON_TYPE_BINARY_OPERATOR /* (op) or (op, others) */ __ARIA_PYTHON_TYPE_BINARY_OPERATOR

//
//
//
#define ARIA_PYTHON_TYPE_END __ARIA_PYTHON_TYPE_END

//
//
//
#define ARIA_PYTHON_ADD_TYPE /* (type) or (type, module) */ __ARIA_PYTHON_ADD_TYPE

} // namespace ARIA
