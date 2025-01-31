#include "ARIA/Buyout.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct SizeOf {
  template <typename T>
  constexpr size_t operator()() const {
    return sizeof(T);
  }
};

} // namespace

TEST(Buyout, Base) {
  constexpr Buyout<SizeOf, int8_t, int16_t, int32_t, int64_t> buyout{SizeOf{}};

  static_assert(buyout.operator()<int8_t>() == 1);
  static_assert(buyout.operator()<int16_t>() == 2);
  static_assert(buyout.operator()<int32_t>() == 4);
  static_assert(buyout.operator()<int64_t>() == 8);

  static_assert(get<int8_t>(buyout) == 1);
  static_assert(get<int16_t>(buyout) == 2);
  static_assert(get<int32_t>(buyout) == 4);
  static_assert(get<int64_t>(buyout) == 8);
}

} // namespace ARIA
