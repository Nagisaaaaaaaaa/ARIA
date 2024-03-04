#pragma once

#include "ARIA/detail/PythonImpl.h"

namespace ARIA {

namespace py = pybind11;

//
//
//
#define ARIA_PYTHON_TYPE_FRIEND __ARIA_PYTHON_TYPE_FRIEND

//
//
//
#define ARIA_PYTHON_TYPE_BEGIN(type) __ARIA_PYTHON_TYPE_BEGIN(type)

#define ARIA_PYTHON_TYPE_METHOD /*(specifiers, name, parameters...)*/                                                  \
  __ARIA_PYTHON_TYPE_METHOD     /*(specifiers, name, parameters...)*/

#define ARIA_PYTHON_TYPE_PROPERTY(name) __ARIA_PYTHON_TYPE_PROPERTY(name)

#define ARIA_PYTHON_TYPE_READONLY_PROPERTY(name) __ARIA_PYTHON_TYPE_READONLY_PROPERTY(name)

#define ARIA_PYTHON_TYPE_UNARY_OPERATOR(op) __ARIA_PYTHON_TYPE_UNARY_OPERATOR(op)

#define ARIA_PYTHON_TYPE_BINARY_OPERATOR(op, others) __ARIA_PYTHON_TYPE_BINARY_OPERATOR(op, others)

#define ARIA_PYTHON_TYPE_END __ARIA_PYTHON_TYPE_END

//
//
//
#define ARIA_ADD_PYTHON_TYPE(type, module) __ARIA_ADD_PYTHON_TYPE(type, module)

} // namespace ARIA
