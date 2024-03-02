#include "ARIA/ARIA.h"

#define SOL_ALL_SAFETIES_ON 1

#include <gtest/gtest.h>
#include <sol/sol.hpp>

namespace ARIA {

TEST(Lua, Base) {
  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::package);

  lua.script("print(\"Hello Lua!\")");

  std::cout << std::endl;
}

} // namespace ARIA
