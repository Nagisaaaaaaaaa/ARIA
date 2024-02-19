#pragma once

#include "ARIA/Property.h"

#include <cuda/std/atomic>

namespace ARIA {

template <typename TThreadUnsafeOrSafe, typename TLabels>
class DisjointSet {
public:
  using value_type = TLabels::value_type;
  static_assert(std::integral<value_type>, "Type of `TLabels` elements should be integral");
  //! Should not use `std::decay_t<decltype(std::declval<TLabels>()[0])>` in order to support proxy systems.

private:
  static_assert(std::is_same_v<TThreadUnsafeOrSafe, ThreadUnsafe> || std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>,
                "`TThreadUnsafeOrSafe` should be substituted with either `ThreadUnsafe` or `ThreadSafe`");
  static constexpr bool threadSafe = std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>;

  //
  //
  //
public:
  DisjointSet() = default;

  ARIA_HOST_DEVICE explicit DisjointSet(const TLabels &labels) : labels_(labels) {}

  ARIA_HOST_DEVICE explicit DisjointSet(TLabels &&labels) : labels_(std::move(labels)) {}

  ARIA_COPY_MOVE_ABILITY(DisjointSet, default, default); //! Let `TLabels` decide.

  //
  //
  //
public:
  /// \brief Get or set the labels.
  ///
  /// \example ```cpp
  /// disjointSet.labels() = ...;
  /// disjointSet.labels()[i] = ...;
  /// ```
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, labels, labels_);

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
  TLabels labels_;
};

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/DisjointSet.inc"
