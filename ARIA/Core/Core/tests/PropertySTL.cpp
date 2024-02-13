#include "ARIA/PropertySTL.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class StringWrapper {
public:
  ARIA_PROP_PREFAB_STD_STRING(public, private, , std::string, str);

private:
  [[nodiscard]] std::string ARIA_PROP_IMPL(str)() const { return "Hello"; }
};

} // namespace

TEST(PropertySTL, String) {
  StringWrapper sw;

  // Capacity.
  EXPECT_TRUE(sw.str().size() == 5);
}

} // namespace ARIA
