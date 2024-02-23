#pragma once

/// \file
/// \warning `BVH` is under developing, interfaces are currently very unstable.
//
//
//
//
//
#include "ARIA/AABB.h"
#include "ARIA/Invocations.h"
#include "ARIA/Launcher.h"

#include <thrust/device_vector.h>

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
// Compute AABB for the given positions.
namespace bvh::detail {

template <typename T>
[[nodiscard]] AABB3<T> ComputeAABB(const thrust::device_vector<Vec3<T>> &positions) {
  auto opVecMin = [] ARIA_HOST_DEVICE(const Vec3<T> &v0, const Vec3<T> &v1) -> Vec3<T> {
    return {min(v0.x(), v1.x()), min(v0.y(), v1.y()), min(v0.z(), v1.z())};
  };
  auto opVecMax = [] ARIA_HOST_DEVICE(const Vec3<T> &v0, const Vec3<T> &v1) -> Vec3<T> {
    return {max(v0.x(), v1.x()), max(v0.y(), v1.y()), max(v0.z(), v1.z())};
  };

  Vec3<T> inf =
      thrust::reduce(positions.begin(), positions.end(), Vec3<T>{infinity<T>, infinity<T>, infinity<T>}, opVecMin);
  Vec3<T> sup =
      thrust::reduce(positions.begin(), positions.end(), Vec3<T>{-infinity<T>, -infinity<T>, -infinity<T>}, opVecMax);

  return AABB3<T>{inf, sup};
}

} // namespace bvh::detail

//
//
//
template <typename TPrimitives, typename FPrimitiveToPos, typename FPrimitiveToAABB>
[[nodiscard]] bool
make_bvh_device(TPrimitives &&primitives, FPrimitiveToPos &&fPrimitiveToPos, FPrimitiveToAABB &&fPrimitiveToAABB) {
  //! Check overflow.
  if (static_cast<uint64>(primitives.size()) > 0x7FFFFFFFLLU)
    ARIA_THROW(
        std::runtime_error,
        "Number of primitives given to the BVH excesses 0x7FFFFFFF, which can not be handled by Karras's algorithm");

  //! Check and determine types.
  using UPrimitivePos = decltype(invoke_with_parentheses_or_brackets(
      fPrimitiveToPos, invoke_with_parentheses_or_brackets(primitives, 0)));
  using UPrimitiveAABB = decltype(invoke_with_parentheses_or_brackets(
      fPrimitiveToAABB, invoke_with_parentheses_or_brackets(primitives, 0)));

  static_assert(vec::detail::is_vec_v<UPrimitivePos>,
                "Type of invocation results of `FPrimitiveToPos` should be `Vec`");
  static_assert(aabb::detail::is_aabb_v<UPrimitiveAABB>,
                "Type of invocation results of `FPrimitiveToAABB` should be `AABB`");

  using UPrimitivePosValueType = std::decay_t<decltype(std::declval<UPrimitivePos>()[0])>;
  using UPrimitiveAABBValueType = std::decay_t<decltype(std::declval<UPrimitiveAABB>().inf()[0])>;

  static_assert(std::is_same_v<UPrimitivePosValueType, UPrimitiveAABBValueType>,
                "Non-consistent types of `Vec` and `AABB`");

  using TReal = UPrimitivePosValueType;
  using TVec = UPrimitivePos;
  using TAABB = UPrimitiveAABB;

  //! Map primitives to positions.
  uint nPrimitives = primitives.size();
  thrust::device_vector<TVec> positions(nPrimitives);

  Launcher(nPrimitives, [=, ps = positions.data()] ARIA_DEVICE(uint i) {
    // Pseudocode: `ps[i] = fPrimitiveToPos[primitives[i]];`.
    ps[i] = invoke_with_parentheses_or_brackets(fPrimitiveToPos, invoke_with_parentheses_or_brackets(primitives, i));
  }).Launch();

  TAABB positionsAABB = bvh::detail::ComputeAABB(positions);

  // TODO: Compute Morton codes, sort by Morton codes, and create the reordering indices.
  //! Never explicitly reorder the primitives.

  // TODO: Add abstractions for CUDA streams.
  cuda::device::current::get().synchronize();

  return true;
}

} // namespace ARIA
