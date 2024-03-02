#include "ARIA/Property.h"
#include "ARIA/PropertySTL.h"

#define SOL_ALL_SAFETIES_ON 1

#include <gtest/gtest.h>
#include <sol/sol.hpp>

namespace ARIA {

namespace {

class Object {
public:
  [[nodiscard]] const std::string &name0() const { return name_; }

  ARIA_PROP_PREFAB_STD_STRING(public, public, , std::string, name1);

public:
  [[nodiscard]] std::vector<int> oneTwoThree() const { return {1, 2, 3}; }

private:
  std::string name_ = "Lua です喵"; // Test UTF-8.

  std::string ARIA_PROP_IMPL(name1)() const { return name_; }

  std::string ARIA_PROP_IMPL(name1)(const std::string &name) { name_ = name; }
};

} // namespace

//
//
//
//
//
TEST(Lua, Base) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  lua.new_usertype<Object>("Object", "name0", &Object::name0, "name1",
                           static_cast<decltype(std::declval<Object>().name1()) (Object::*)()>(&Object::name1),
                           "oneTwoThree", &Object::oneTwoThree);

  lua.new_usertype<decltype(std::declval<Object>().name1())>(
      "Object::name1", "value",
      static_cast<decltype(std::declval<Object>().name1().value()) (decltype(std::declval<Object>().name1())::*)()>(
          &decltype(std::declval<Object>().name1())::value));

  Object obj;
  lua["obj"] = &obj;

  lua.script("local name0 = obj:name0()"
             "local name1 = obj:name1()"
             ""
             "assert(name0 == \"Lua です喵\", \"Error message です喵\")"
             "assert(tostring(name1) == \"Lua です喵\", \"Error message です喵\")"
             "assert(name1:value() == \"Lua です喵\", \"Error message です喵\")"
             "assert(obj:oneTwoThree()[1] == 1)"
             "assert(obj:oneTwoThree()[2] == 2)"
             "assert(obj:oneTwoThree()[3] == 3)");
}

} // namespace ARIA
