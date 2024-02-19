#pragma once

#include "ARIA/ARIA.h"

namespace ARIA {

template <typename TThreadSafeOrNot, typename TLabels>
class DisjointSet {
public:
  DisjointSet() = default;

  DisjointSet(const TLabels &labels) : labels_(labels) {}

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
