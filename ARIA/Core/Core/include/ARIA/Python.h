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

namespace py = python::detail::py;

//
//
//
using python::detail::ScopedInterpreter;

using python::detail::Module;

using python::detail::Dict;

//
//
//
#define ARIA_PYTHON_TYPE_FRIEND __ARIA_PYTHON_TYPE_FRIEND

//
//
//
#define ARIA_PYTHON_TYPE_BEGIN(type) __ARIA_PYTHON_TYPE_BEGIN(type)

#define ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_) __ARIA_PYTHON_TEMPLATE_TYPE_BEGIN(template_)

#define ARIA_PYTHON_TYPE_METHOD /*(specifiers, name, parameters...)*/ __ARIA_PYTHON_TYPE_METHOD

#define ARIA_PYTHON_TYPE_PROPERTY(name) __ARIA_PYTHON_TYPE_PROPERTY(name)

#define ARIA_PYTHON_TYPE_READONLY_PROPERTY(name) __ARIA_PYTHON_TYPE_READONLY_PROPERTY(name)

#define ARIA_PYTHON_TYPE_UNARY_OPERATOR(op) __ARIA_PYTHON_TYPE_UNARY_OPERATOR(op)

#define ARIA_PYTHON_TYPE_BINARY_OPERATOR /*(op) or (op, others)*/ __ARIA_PYTHON_TYPE_BINARY_OPERATOR

#define ARIA_PYTHON_TYPE_END __ARIA_PYTHON_TYPE_END

//
//
//
#define ARIA_ADD_PYTHON_TYPE(type, module) __ARIA_ADD_PYTHON_TYPE(type, module)

} // namespace ARIA
