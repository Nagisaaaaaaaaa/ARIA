#include "ARIA/MovingPoint.h"
#include "ARIA/Constant.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename T, auto dim, typename degree>
class BezierCurve {
public:
  bool IsInDomain(const T &t) const {}

  const Vec<T, dim> &operator()(const T &t) const {}
};

} // namespace

TEST(MovingPoint, Base) {
  static_assert(MovingPoint<BezierCurve, Real, 2, C<5>>);
}

} // namespace ARIA
