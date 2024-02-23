#pragma once

/// \file
/// \warning `BVH` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/ARIA.h"

namespace ARIA {

template <typename TSpace, typename TPrimitives>
class BVH;

template <typename TPrimitives>
class BVH<SpaceDevice, TPrimitives> {
public:
private:
};

//
//
//
//
//
template <typename TPrimitives, typename FPrimitiveToPos, typename FPrimitiveToAABB>
[[nodiscard]] auto
make_bvh_device(TPrimitives &&primitives, FPrimitiveToPos &&fPrimitiveToPos, FPrimitiveToAABB &&fPrimitiveToAABB) {
  //! 1. Check safety.
  if (primitives.size() > 0x7FFFFFFFU)
    ARIA_THROW(
        std::runtime_error,
        "Number of primitives given to the BVH excesses 0x7FFFFFFF, which can not be handled by Karras's algorithm");

  // TODO: Compute positions of every primitives.
  // TODO: Compute AABB of all the positions.
  // TODO: Compute Morton codes, sort by Morton codes, and create the reordering indices.
  //! Never explicitly reorder the primitives.
}

} // namespace ARIA
