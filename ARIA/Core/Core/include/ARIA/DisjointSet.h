#pragma once

#include "ARIA/ARIA.h"

#include <cuda/std/atomic>

namespace ARIA {

template <typename TThreadUnsafeOrSafe, typename TLabels>
class DisjointSet;

template <typename TLabels>
class DisjointSet<ThreadSafe, TLabels> {
public:
  DisjointSet() = default;

  explicit DisjointSet(const TLabels &labels) : labels_(labels) {}

public:
  template <typename Coord>
  static decltype(auto) Find(const Coord &coord) {}

  template <typename Coord>
  static decltype(auto) FindAndCompress(const Coord &coord) {}

  template <typename Coord>
  static decltype(auto) Union(const Coord &coord0, const Coord &coord1) {}

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
