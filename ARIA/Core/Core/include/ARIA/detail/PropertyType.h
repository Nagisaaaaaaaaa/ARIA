#pragma once

#include "ARIA/ARIA.h"

#include <concepts>

namespace ARIA {

namespace property::detail {

/// \brief The base class of any property.
/// The base class is introduced in order to
/// automatically define operators for properties with CRTP and type checks.
///
/// See PropertyImpl.h for its implementation.
template <typename TProperty>
class PropertyBase;

/// \brief Whether the type `T` is a property.
template <typename T>
concept PropertyType = std::derived_from<T, property::detail::PropertyBase<T>>;

/// \brief Whether the type `T` is not a property.
template <typename T>
concept NonPropertyType = !PropertyType<T>;

} // namespace property::detail

} // namespace ARIA
