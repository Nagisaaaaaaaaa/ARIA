#pragma once

#include "ARIA/ARIA.h"

namespace ARIA {

// Fwd.
class Object;
class Component;

//
//
//
/// \brief Removes a `Object`.
/// Actual destruction is always delayed until after the current event loop,
/// but is always done before rendering.
void Destroy(Object &object);

/// \brief Removes a `Component`.
/// Actual destruction is always delayed until after the current event loop,
/// but is always done before rendering.
void Destroy(Component &component);

//
//
//
/// \brief Perform the actual destruction for all previous destroyed items during
/// the current event loop.
void DestructAllDestroyed();

} // namespace ARIA
