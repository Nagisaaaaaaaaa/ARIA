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
#include "ARIA/MortonCode.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

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
namespace bvh::detail {

// Compute AABB for the given positions.
template <typename T>
[[nodiscard]] AABB3<T> ComputeAABB(const thrust::device_vector<Vec3<T>> &positionsD) {
  auto opVecMin = [] ARIA_HOST_DEVICE(const Vec3<T> &v0, const Vec3<T> &v1) -> Vec3<T> {
    return {min(v0.x(), v1.x()), min(v0.y(), v1.y()), min(v0.z(), v1.z())};
  };
  auto opVecMax = [] ARIA_HOST_DEVICE(const Vec3<T> &v0, const Vec3<T> &v1) -> Vec3<T> {
    return {max(v0.x(), v1.x()), max(v0.y(), v1.y()), max(v0.z(), v1.z())};
  };

  Vec3<T> inf =
      thrust::reduce(positionsD.begin(), positionsD.end(), Vec3<T>{infinity<T>, infinity<T>, infinity<T>}, opVecMin);
  Vec3<T> sup =
      thrust::reduce(positionsD.begin(), positionsD.end(), Vec3<T>{-infinity<T>, -infinity<T>, -infinity<T>}, opVecMax);

  return AABB3<T>{inf, sup};
}

// Sort `positionsD` by Morton codes.
template <typename T>
void SortByMortonCodes(const thrust::device_vector<Vec3<T>> &positionsD,
                       thrust::device_vector<uint> &sortedIndicesD,
                       thrust::device_vector<uint64> &sortedMortonCodesD) {
  // Check safety.
  auto nPositions = Auto(positionsD.size());

  if ((sortedIndicesD.size() != 0 && sortedIndicesD.size() != nPositions) ||
      (sortedMortonCodesD.size() != 0 && sortedMortonCodesD.size() != nPositions))
    ARIA_THROW(std::runtime_error, "Inconsistent input");

  // Compute AABB.
  AABB3<T> aabb = ComputeAABB(positionsD);

  // Allocate and init.
  if (sortedIndicesD.empty()) { // Init to [0, 1, 2, ... ] if indices have not been initialized.
    sortedIndicesD.resize(nPositions);
    Launcher(nPositions, [sortedIndices = sortedIndicesD.data()] ARIA_DEVICE(size_t i) {
      sortedIndices[i] = i;
    }).Launch();
  }

  sortedMortonCodesD.resize(nPositions);

  // Compute Morton codes.
  Launcher(nPositions,
           [=, positions = positionsD.data(), sortedMortonCodes = sortedMortonCodesD.data()] ARIA_DEVICE(size_t i) {
    Vec3<T> offset = aabb.offset(positions[i]);
    Vec3u coord = (offset * T{1 << 21}).template cast<uint>(); // TODO: Magic number.
    sortedMortonCodes[i] = MortonCode<3>::Encode(coord);
  });

  // Sort indices by Morton codes.
  thrust::sort_by_key(sortedMortonCodesD.begin(), sortedMortonCodesD.end(), sortedIndicesD.begin());
}

template <typename T>
void BuildKarras1ByMortonCodes(const ArrayD<Vector3f> &vertices_d,
                               const ArrayD<Vector3f> &normals_d,
                               const ArrayD<Vector3u> &triangles_d,
                               const ArrayD<uint16_t> &triangle_indices_to_object_indices_d,
                               const ArrayD<uint64_t> &morton_codes_d,
                               NodesVariant &nodes_d,
                               TessellatedBinaryBVHDataType3f &node_type,
                               ArrayD<Family> &families_d,
                               ArrayD<unsigned> &interior_atomics_d) {
  unsigned num_triangles = triangles_d.Size();

  // build tree structure
  CK_MUST(nodes_d.Realloc(2 * num_triangles - 1));
  CK_MUST(families_d.Realloc(2 * num_triangles - 1));
  CK_MUST(interior_atomics_d.Realloc(num_triangles - 1));

  auto vertices = vertices_d.Span();
  auto triangles = triangles_d.Span();
  auto morton_codes = morton_codes_d.Span();
  auto nodes = nodes_d.Span();
  auto families = families_d.Span();
  auto interior_atomics = interior_atomics_d.Span();

  // build interior nodes
  CK_MUST(cudaMemcpy(&families_d[0], &root_family, sizeof(Family), cudaMemcpyHostToDevice));

  CK_MUST(CUDALaunch::ForEach(num_triangles - 1, [=] DEV(int i) mutable {
    auto code_i = morton_codes[i];

    auto delta = [&](int j) -> int {
      if (j < 0 || j >= int(num_triangles))
        return -1;
      else {
        auto code_j = morton_codes[j];
        if (code_i == code_j)
          return 64 + Clz(uint32_t(i ^ j));
        else
          return Clz(uint64_t(code_i ^ code_j));
      }
    };

    // determine direction of the range
    auto delta_i_iadd1 = delta(i + 1);
    auto delta_i_isub1 = delta(i - 1);
    int d, delta_min;
    if (delta_i_iadd1 - delta_i_isub1 > 0) {
      d = 1;
      delta_min = delta_i_isub1;
    } else {
      d = -1;
      delta_min = delta_i_iadd1;
    }

    // compute upper bound for the length of the range
    int l_max = 2;
    while (delta(i + l_max * d) > delta_min)
      l_max <<= 1;

    // find the other end using binary search
    int l = 0;
    for (int t = l_max >> 1; t > 0; t >>= 1)
      if (delta(i + (l + t) * d) > delta_min)
        l += t;
    auto j = i + l * d;

    int il, ir;
    if (d > 0)
      il = i, ir = j;
    else
      il = j, ir = i;

    // find the split position using binary search
    auto delta_node = delta(j);
    int s = 0;
    for (int div = 2, t; (t = (l + div - 1) / div) > 0; div <<= 1)
      if (delta(i + (s + t) * d) > delta_node)
        s += t;
    auto gamma = i + s * d + min(d, 0);

    // setup data
    bool is_interior_l, is_interior_r;

    if (il == gamma) { // &bvh_leaves[gamma]
      is_interior_l = false;
      auto i_leaf = num_triangles - 1 + gamma;
      families[i_leaf].Set<Family::child_l>(i);
    } else { // &bvh_interiors[gamma]
      is_interior_l = true;
      auto i_interior = gamma;
      families[i_interior].Set<Family::child_l>(i);
    }
    if (ir == gamma + 1) { // &bvh_leaves[gamma + 1]
      is_interior_r = false;
      auto i_leaf = num_triangles + gamma;
      families[i_leaf].Set<Family::child_r>(i);
    } else { // &bvh_interiors[gamma + 1]
      is_interior_r = true;
      auto i_interior = gamma + 1;
      families[i_interior].Set<Family::child_r>(i);
    }

    nodes[i].data_karras1.SetInterior(gamma, is_interior_r, is_interior_l);
  }));

  // build leaf nodes
  CK_MUST(CUDALaunch::ForEach(num_triangles, [=] DEV(unsigned i) mutable {
    auto i_leaf = num_triangles - 1 + i;

    // setup data
    nodes[i_leaf].data_karras1.SetLeaf();
  }));
  node_type = TessellatedBinaryBVHDataType3f::karras1;

  // compute aabbs and metadata
  CK_MUST(cudaMemset(interior_atomics_d.Get(), 0, sizeof(unsigned) * interior_atomics_d.Size()));

  CK_MUST(CUDALaunch::ForEach(num_triangles, [=] DEV(unsigned i) mutable {
    auto i_curr = num_triangles - 1 + i;

    // compute leaf bound
    auto leaf_triangle = triangles[i];
    AABB3f leaf_bound(vertices[leaf_triangle.x], vertices[leaf_triangle.y], vertices[leaf_triangle.z]);
    nodes[i_curr].AABB(leaf_bound);

    // from leaf to root
    while (true) {
      auto i_parent = families[i_curr].Parent();

      // check whether traversal should stop or not
      __threadfence();
      if (i_parent == root_parent || atomicAdd(interior_atomics.data() + i_parent, 1) ==
                                         0) { // stop if is root or firstly access an internal parent
        break;
      } else { // continue if secondly access an internal parent
        auto data = nodes[i_parent].data_karras1;
        auto left_interior = data.LeftInterior();
        auto left = data.Left(left_interior, num_triangles - 1);
        auto right = data.Right(left_interior, num_triangles - 1);

        // compute interior bound
        AABB3f parent_bound(nodes[left].AABB(), nodes[right].AABB());
        nodes[i_parent].AABB(parent_bound);

        // iterate
        i_curr = i_parent;
      }
    }
  }));
}

} // namespace bvh::detail

//
//
//
template <typename TPrimitives, typename FPrimitiveToPos, typename FPrimitiveToAABB>
[[nodiscard]] bool
make_bvh_device(TPrimitives &&primitives, FPrimitiveToPos &&fPrimitiveToPos, FPrimitiveToAABB &&fPrimitiveToAABB) {
  // Check overflow.
  if (static_cast<uint64>(primitives.size()) > 0x7FFFFFFFLLU)
    ARIA_THROW(
        std::runtime_error,
        "Number of primitives given to the BVH excesses 0x7FFFFFFF, which can not be handled by Karras's algorithm");

  // Check and determine types.
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

  // Map primitives to positions.
  uint nPrimitives = primitives.size();
  thrust::device_vector<TVec> positionsD(nPrimitives);

  Launcher(nPrimitives, [=, positions = positionsD.data()] ARIA_DEVICE(uint i) {
    // Pseudocode: `positions[i] = fPrimitiveToPos[primitives[i]];`.
    positions[i] =
        invoke_with_parentheses_or_brackets(fPrimitiveToPos, invoke_with_parentheses_or_brackets(primitives, i));
  }).Launch();

  // Compute Morton codes, sort by Morton codes, and create the reordering indices.
  thrust::device_vector<uint> sortedIndicesD;
  thrust::device_vector<uint64> sortedMortonCodesD;
  bvh::detail::SortByMortonCodes(positionsD, sortedIndicesD, sortedMortonCodesD);
  auto sortedPrimitives = [=, sortedIndices = sortedIndicesD.data()] ARIA_HOST_DEVICE(uint i) {
    return invoke_with_parentheses_or_brackets(primitives, sortedIndicesD[i]);
  };

  //! Never explicitly reorder the primitives.

  // Perform Karras's algorithm.
  bvh::detail::BuildKarras1ByMortonCodes();

  // TODO: Add abstractions for CUDA streams.
  cuda::device::current::get().synchronize();

  return true;
}

} // namespace ARIA
