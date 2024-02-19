#pragma once

#include "ARIA/Property.h"

#include <cuda/std/atomic>

namespace ARIA {

template <typename TThreadUnsafeOrSafe, typename TLabels>
class DisjointSet;

template <typename TLabels>
class DisjointSet<ThreadSafe, TLabels> {
private:
  using TLabel = TLabels::value_type;
  static_assert(std::integral<TLabel>, "Type of labels should be integral");
  //! Should not use `std::decay_t<decltype(std::declval<TLabels>()[0])>` in order to support proxy systems.

public:
  DisjointSet() = default;

  explicit DisjointSet(const TLabels &labels) : labels_(labels) {}

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, labels, labels_);

public:
  ARIA_HOST_DEVICE TLabel Find(const TLabel &i) const {
    TLabel new_i;

    while ((new_i = labels()[i]) != i) {
      i = new_i;
    }

    return i;
  }

  ARIA_HOST_DEVICE TLabel FindAndCompress(const TLabel &i) {
    TLabel i_c = i;

    size_t new_i;

    while ((new_i = labels()[i]) != i) {
      i = new_i;
      labels()[i_c] = i;
    }

    return i;
  }

  ARIA_HOST_DEVICE TLabel Union(const TLabel &i0, const TLabel &i1) {
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
