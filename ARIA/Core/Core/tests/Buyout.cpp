#include "ARIA/Buyout.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

struct printTypeName {
  template <typename T>
  std::string operator()() const {
    return typeid(T).name();
  }
};

} // namespace

TEST(Buyout, Base) {
  Buyout<std::string, printTypeName, int, float, double> buyout{printTypeName{}};

  EXPECT_EQ(buyout.operator()<int>(), "int");
  EXPECT_EQ(buyout.operator()<float>(), "float");
  EXPECT_EQ(buyout.operator()<double>(), "double");

  EXPECT_EQ(get<int>(buyout), "int");
  EXPECT_EQ(get<float>(buyout), "float");
  EXPECT_EQ(get<double>(buyout), "double");
}

} // namespace ARIA
