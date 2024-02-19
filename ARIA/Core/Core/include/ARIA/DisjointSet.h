#pragma once

/// \file
/// \brief A policy-based disjoint set implementation.
///
/// In computer science, a disjoint-set data structure, also called
/// a union–find data structure or merge–find set, is a data structure that
/// stores a collection of disjoint (non-overlapping) sets.
/// Equivalently, it stores a partition of a set into disjoint subsets.
/// It provides operations for merging sets (see `Union()`), and
/// finding a representative member of a set (see `Find()` and `FindAndCompress()`).
/// The last operation makes it possible to find out efficiently if
/// any two elements are in the same or different sets.
//
//
//
//
//
#include "ARIA/Property.h"

#include <cuda/std/atomic>

namespace ARIA {

/// \brief A policy-based disjoint set implementation.
///
/// In computer science, a disjoint-set data structure, also called
/// a union–find data structure or merge–find set, is a data structure that
/// stores a collection of disjoint (non-overlapping) sets.
/// Equivalently, it stores a partition of a set into disjoint subsets.
/// It provides operations for merging sets (see `Union()`), and
/// finding a representative member of a set (see `Find()` and `FindAndCompress()`).
/// The last operation makes it possible to find out efficiently if
/// any two elements are in the same or different sets.
///
/// \tparam TThreadUnsafeOrSafe A policy controls whether `Union()` is thread-safe or not.
/// `ThreadUnsafe` or `ThreadSafe` should be substituted here.
/// \tparam TNodes A "node" is the context which a disjoint set element contains,
/// "nodes" is the container of all the disjoint set elements, and `TNodes` is type of this container.
/// For this disjoint set implementation, `TNodes` can be an owning container, such as
/// `std::vector`, `std::array`, `thrust::host_vector`, `thrust::device_vector`, `TensorVector`, .etc.
/// or a non-owning view, such as `std::span`, `Tensor`, .etc.
///
/// \example ```cpp
/// using VolumeHost = TensorVectorHost<Real, C<3>>;
/// using DisjointSetHost = DisjointSet<ThreadSafe, VolumeHost>;
/// DisjointSetHost disjointSet{VolumeHost{make_layout_major(100, 200, 400)}};
/// ```
template <typename TThreadUnsafeOrSafe, typename TNodes>
class DisjointSet {
public:
  using value_type = TNodes::value_type;
  static_assert(std::integral<value_type>, "Type of `TNodes` elements should be integral");
  //! Should not use `std::decay_t<decltype(std::declval<TNodes>()[0])>` in order to support proxy systems.

private:
  static_assert(std::is_same_v<TThreadUnsafeOrSafe, ThreadUnsafe> || std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>,
                "`TThreadUnsafeOrSafe` should be substituted with either `ThreadUnsafe` or `ThreadSafe`");
  static constexpr bool threadSafe = std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>;

  //
  //
  //
public:
  DisjointSet() = default;

  ARIA_HOST_DEVICE explicit DisjointSet(const TNodes &nodes) : nodes_(nodes) {}

  ARIA_HOST_DEVICE explicit DisjointSet(TNodes &&nodes) : nodes_(std::move(nodes)) {}

  ARIA_COPY_MOVE_ABILITY(DisjointSet, default, default); //! Let `TNodes` decide.

  //
  //
  //
public:
  /// \brief Get or set the nodes.
  ///
  /// \example ```cpp
  /// disjointSet.nodes() = ...;
  /// disjointSet.nodes()[i] = ...;
  /// ```
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, nodes, nodes_);

  //
  //
  //
public:
  /// \brief Takes a node of a tree as input and returns its root as output.
  ///
  /// \example ```cpp
  /// int root = disjointSet.Find(100);
  /// ```
  [[nodiscard]] ARIA_HOST_DEVICE value_type Find(value_type i) const;

  /// \brief Takes a node of a tree as input and returns its root as output.
  /// This method differs from `Find` in that it will usually
  /// compress the tree to accelerate later accesses.
  ///
  /// \example ```cpp
  /// int root = disjointSet.FindAndCompress(100);
  /// ```
  ARIA_HOST_DEVICE value_type FindAndCompress(value_type i);

  /// \brief Takes two nodes as inputs and joins together the trees they belong to, by
  /// setting one tree root as the father of the other one.
  ///
  /// \example ```cpp
  /// disjointSet.Union(100, 50);
  /// ```
  ///
  /// \warning In detail, suppose `r0` is the root of `i0` and `r1` is the root of `i1`,
  /// If `r0 < r1`, `r0` is set as the father of `r1`.
  /// If `r1 < r0`, `r1` is set as the father of `r0`.
  /// That is, "the smaller, the father".
  ///
  /// This restriction is introduced to support thread-safe disjoint sets.
  /// Also, to make behavior of the codes consistent,
  /// thread-unsafe disjoint sets also follow this rule.
  ARIA_HOST_DEVICE void Union(value_type i0, value_type i1);

  //
  //
  //
  //
  //
private:
  TNodes nodes_;
};

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/DisjointSet.inc"
