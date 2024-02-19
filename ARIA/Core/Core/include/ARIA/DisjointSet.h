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
  TLabel Find(const TLabel &i) const {
    TLabel new_i;

    while ((new_i = labels()[i]) != i) {
      i = new_i;
    }

    return i;
  }

  TLabel FindAndCompress(const TLabel &i) {}

  TLabel Union(const TLabel &i0, const TLabel &i1) {}

private:
  TLabels labels_;
};

} // namespace ARIA

#if 0
DEV static ALWAYS_INLINE size_t CCLFind(
    const std::span<size_t>& ccl,
    size_t i) {
  size_t new_i;

  while ((new_i = ccl[i]) != i) {
    i = new_i;
  }

  return i;
}

DEV static ALWAYS_INLINE size_t CCLFindAndCompress(
    std::span<size_t>& ccl,
    size_t i) {
  auto i_c = i;

  size_t new_i;

  while ((new_i = ccl[i]) != i) {
    i = new_i;
    ccl[i_c] = i;
  }

  return i;
}

DEV static ALWAYS_INLINE void CCLUnion(
    std::span<size_t>& ccl,
    size_t i0, size_t i1) {
  bool done;

  do {
    i0 = CCLFind(ccl, i0);
    i1 = CCLFind(ccl, i1);

    if (i0 < i1) {
      auto old = atomicMin(&ccl[i1], i0);
      done = (old == i1);
      i1 = old;
    }
    else if (i1 < i0) {
      auto old = atomicMin(&ccl[i0], i1);
      done = (old == i0);
      i0 = old;
    }
    else {
      done = true;
    }
  } while (!done);
}
#endif
