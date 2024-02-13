#pragma once

/// \file
/// \brief This file defines property and sub-property prefabs for
/// commonly used STL types, such as `std::vector` and `std::string`, in order to
/// make it easier to define properties with these types.

//
//
//
//
//
#include "ARIA/detail/PropertySTLImpl.h"

namespace ARIA {

/// \brief A property prefab for `std::string`.
/// All possible sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_PROP_PREFAB_STD_STRING(public, public, __host__, std::string, name);
/// ```
#define ARIA_PROP_PREFAB_STD_STRING(accessGet, accessSet, specifiers, type, propName)                                  \
  __ARIA_PROP_PREFAB_STD_STRING(accessGet, accessSet, specifiers, type, propName)

/// \brief A sub-property prefab for `std::string`.
/// All possible sub-sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_SUB_PROP_PREFAB_STD_STRING(__host__ , std::string, name);
/// ```
#define ARIA_SUB_PROP_PREFAB_STD_STRING(specifiers, type, propName)                                                    \
  __ARIA_SUB_PROP_PREFAB_STD_STRING(specifiers, type, propName)

} // namespace ARIA
