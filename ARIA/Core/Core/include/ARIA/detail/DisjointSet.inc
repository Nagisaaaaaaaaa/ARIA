#pragma once

namespace ARIA {

template <typename U0, typename U1>
ARIA_HOST_DEVICE DisjointSet<U0, U1>::value_type DisjointSet<U0, U1>::Find(DisjointSet<U0, U1>::value_type i) const {
  value_type iNew;

  while ((iNew = labels()[i]) != i) {
    i = iNew;
  }

  return i;
}

template <typename U0, typename U1>
ARIA_HOST_DEVICE DisjointSet<U0, U1>::value_type
DisjointSet<U0, U1>::FindAndCompress(DisjointSet<U0, U1>::value_type i) {
  value_type iCpy = i;

  size_t iNew;

  while ((iNew = labels()[i]) != i) {
    i = iNew;
    labels()[iCpy] = i;
  }

  return i;
}

template <typename U0, typename TLabels>
ARIA_HOST_DEVICE void DisjointSet<U0, TLabels>::Union(DisjointSet<U0, TLabels>::value_type i0,
                                                      DisjointSet<U0, TLabels>::value_type i1) {
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
        static_assert(std::is_same_v<value_type, decltype(Auto(std::declval<TLabels>()[0]))>,
                      "Return type of `TLabels::operator[]` should represents the same type with `value_type`");
        static_assert(std::is_reference_v<decltype(std::declval<TLabels>()[0])>,
                      "Return type of `TLabels::operator[]` should be a reference");
        //! Proxy systems are not allowed here because constructors of `atomic_ref` only accept references.

        cuda::atomic_ref label1{labels()[i1]};
        auto old = label1.fetch_min(i0);
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
      labels()[i1] = i0;
  }
}

} // namespace ARIA
