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

public:
  DisjointSet() = default;

  ARIA_HOST_DEVICE explicit DisjointSet(const TLabels &labels) : labels_(labels) {}

  ARIA_HOST_DEVICE explicit DisjointSet(TLabels &&labels) : labels_(std::move(labels)) {}

  ARIA_COPY_MOVE_ABILITY(DisjointSet, default, default); //! Let `TLabels` decide.

public:
  /// \brief Get or set the labels.
  ///
  /// \example ```cpp
  /// disjointSet.labels() = ...;
  /// disjointSet.labels()[i] = ...;
  /// ```
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, labels, labels_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE value_type Find(value_type i) const;

  ARIA_HOST_DEVICE value_type FindAndCompress(value_type i);

  ARIA_HOST_DEVICE void Union(value_type i0, value_type i1);

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
