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
                "Type of `TThreadUnsafeOrSafe` should be either `ThreadUnsafe` or `ThreadSafe`");
  static constexpr bool threadSafe = std::is_same_v<TThreadUnsafeOrSafe, ThreadSafe>;

public:
  DisjointSet() = default;

  ARIA_HOST_DEVICE explicit DisjointSet(const TLabels &labels) : labels_(labels) {}

  ARIA_COPY_MOVE_ABILITY(DisjointSet, default, default); //! Let `TLabels` decide.

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, labels, labels_);

public:
  ARIA_HOST_DEVICE value_type Find(const value_type &i) const {
    value_type new_i;

    while ((new_i = labels()[i]) != i) {
      i = new_i;
    }

    return i;
  }

  ARIA_HOST_DEVICE value_type FindAndCompress(const value_type &i) {
    value_type i_c = i;

    size_t new_i;

    while ((new_i = labels()[i]) != i) {
      i = new_i;
      labels()[i_c] = i;
    }

    return i;
  }

  ARIA_HOST_DEVICE value_type Union(const value_type &i0, const value_type &i1) {
    bool done;

    do {
      i0 = Find(i0);
      i1 = Find(i1);

      if (i1 < i0) {
        using std::swap;
        swap(i0, i1);
      }

      if (i0 < i1) {
        cuda::atomic_ref label1{labels()[i1]};
        auto old = label1.fetch_min(i0);
        done = (old == i1);
        i1 = old;
      } else { // i0 == i1.
        done = true;
      }
    } while (!done);
  }

private:
  TLabels labels_;
};

} // namespace ARIA
