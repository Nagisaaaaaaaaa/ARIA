#pragma once

namespace ARIA {

//! This implementation is based on
//! 2018, Stefano Allegretti, .etc, Optimizing GPU-Based Connected Components Labelling Algorithms.
//!
//! Since the paper has explained the algorithms in detail, this file is not documented.
//! Please read the paper if you want to understand the codes.

template <typename U0, typename U1>
template <typename... Coords>
ARIA_HOST_DEVICE inline DisjointSet<U0, U1>::value_type DisjointSet<U0, U1>::Find(Coords &&...coords) const {
  value_type i;

  if constexpr (sizeof...(Coords) == 1 && (std::convertible_to<Coords, value_type> && ...))
    i = static_cast<value_type>(std::get<0>(std::forward_as_tuple(std::forward<Coords>(coords)...)));
  else if constexpr (sizeof...(Coords) == 1)
    i = nodes()[std::get<0>(std::forward_as_tuple(std::forward<Coords>(coords)...))];
  else
    i = nodes()(std::forward<Coords>(coords)...);

  value_type iNew;

  while ((iNew = nodes()[i]) != i) {
    i = iNew;
  }

  return i;
}

template <typename U0, typename U1>
template <typename... Coords>
ARIA_HOST_DEVICE inline DisjointSet<U0, U1>::value_type DisjointSet<U0, U1>::FindAndCompress(Coords &&...coords) {
  value_type i;

  if constexpr (sizeof...(Coords) == 1 && (std::convertible_to<Coords, value_type> && ...))
    i = static_cast<value_type>(std::get<0>(std::forward_as_tuple(std::forward<Coords>(coords)...)));
  else if constexpr (sizeof...(Coords) == 1)
    i = nodes()[std::get<0>(std::forward_as_tuple(std::forward<Coords>(coords)...))];
  else
    i = nodes()(std::forward<Coords>(coords)...);

  size_t iNew;

  while ((iNew = nodes()[i]) != i) {
    i = iNew;

    // Pseudocode: `nodes()[i] = i`.
    if constexpr (sizeof...(Coords) == 1)
      nodes()[std::get<0>(std::forward_as_tuple(std::forward<Coords>(coords)...))] = i;
    else
      nodes()(std::forward<Coords>(coords)...) = i;
  }

  return i;
}

template <typename U0, typename TNodes>
template <typename Coords>
ARIA_HOST_DEVICE inline void DisjointSet<U0, TNodes>::Union(const Coords &coords0, const Coords &coords1) {
  value_type i0, i1;

  if constexpr (std::convertible_to<Coords, value_type>) {
    i0 = static_cast<value_type>(coords0);
    i1 = static_cast<value_type>(coords1);
  } else {
    i0 = nodes()[coords0];
    i1 = nodes()[coords1];
  }

  if constexpr (threadSafe) {
    bool done;

    do {
      i0 = Find(i0);
      i1 = Find(i1);

      if (i1 < i0) {
        using std::swap;
        swap(i0, i1);
      }

      if (i0 < i1) {
        static_assert(std::is_reference_v<decltype(std::declval<TNodes>()[0])>,
                      "Return type of `TNodes::operator[]` should be a reference");
        //! Proxy systems are not allowed here because constructors of `atomic_ref` only accept references.

        cuda::atomic_ref node1{nodes()[i1]};
        auto old = node1.fetch_min(i0);
        done = (old == i1);
        i1 = old;
      } else { // i0 == i1.
        done = true;
      }
    } while (!done);
  } else {
    i0 = Find(i0);
    i1 = Find(i1);

    if (i1 < i0) {
      using std::swap;
      swap(i0, i1);
    }

    if (i0 < i1)
      nodes()[i1] = i0;
  }
}

template <typename U0, typename U1>
template <typename... Coords>
  requires(sizeof...(Coords) > 1)
ARIA_HOST_DEVICE inline void DisjointSet<U0, U1>::Union(Coords &&...coords0, Coords &&...coords1) {
  return Union(nodes()(std::forward<Coords>(coords0)...), nodes()(std::forward<Coords>(coords1)...));
}

} // namespace ARIA
