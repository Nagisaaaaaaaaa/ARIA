#pragma once

#include "ARIA/Property.h"

#include <pybind11/embed.h>
#include <pybind11/operators.h>

namespace ARIA {

namespace py = pybind11;

template <typename T>
void DefinePythonType(const py::module_ &module);

//
//
//
#define ARIA_PYTHON_TYPE_FRIEND                                                                                        \
                                                                                                                       \
  template <typename TUVW>                                                                                             \
  friend void ::ARIA::DefinePythonType(const py::module_ &module)

//
//
//
#define ARIA_PYTHON_TYPE_BEGIN(TYPE)                                                                                   \
                                                                                                                       \
  template <>                                                                                                          \
  void DefinePythonType<TYPE>(const py::module_ &module) {                                                             \
    using Type = TYPE;                                                                                                 \
                                                                                                                       \
    py::class_<Type> cls(module, #TYPE)

//
//
//
// clang-format off
#define __ARIA_PYTHON_TYPE_METHOD_PARAMS2(SPECIFIERS, NAME)                                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  ))                                                                                                                   \
  (Type::*)() SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS3(SPECIFIERS, NAME, T0)                                                        \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>()))                                                                                                 \
  (Type::*)(T0) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS4(SPECIFIERS, NAME, T0, T1)                                                    \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (Type::*)(T0, T1) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS5(SPECIFIERS, NAME, T0, T1, T2)                                                \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (Type::*)(T0, T1, T2) SPECIFIERS>(&Type::NAME))

#define __ARIA_PYTHON_TYPE_METHOD_PARAMS6(SPECIFIERS, NAME, T0, T1, T2, T3)                                            \
  cls.def(#NAME, static_cast<decltype(std::declval<SPECIFIERS Type>().NAME(                                            \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (Type::*)(T0, T1, T2, T3) SPECIFIERS>(&Type::NAME))
// clang-format on

#define __ARIA_PYTHON_TYPE_METHOD(...)                                                                                 \
  __ARIA_EXPAND(__ARIA_EXPAND(ARIA_CONCAT(__ARIA_PYTHON_TYPE_METHOD_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

#define ARIA_PYTHON_TYPE_METHOD __ARIA_PYTHON_TYPE_METHOD

//
//
//
#define ARIA_PYTHON_TYPE_PROPERTY(NAME)                                                                                \
                                                                                                                       \
  cls.def_property(#NAME, static_cast<decltype(std::declval<Type>().NAME()) (Type::*)()>(&Type::NAME),                 \
                   static_cast<void (Type::*)(const decltype(std::declval<Type>().ARIA_PROP_IMPL(NAME)()) &)>(         \
                       &Type::ARIA_PROP_IMPL(NAME)))

//
//
//
#define ARIA_PYTHON_TYPE_END                                                                                           \
  }                                                                                                                    \
  using ARIA::DefinePythonType

} // namespace ARIA
