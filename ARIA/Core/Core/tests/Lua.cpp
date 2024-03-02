#include "ARIA/Property.h"
#include "ARIA/PropertySTL.h"

#define SOL_ALL_SAFETIES_ON 1

#include <gtest/gtest.h>
#include <sol/sol.hpp>

namespace ARIA {

namespace {

class GrandParent {
public:
  virtual ~GrandParent() = default;

  virtual int value() = 0;
};

class Parent : public GrandParent {
public:
  virtual ~Parent() = default;

  int value() override { return 1; }
};

class Child final : public Parent {
public:
  virtual ~Child() = default;

  int value() final { return 2; }
};

//
//
//
class Object {
public:
  [[nodiscard]] const std::string &name0() const { return name_; }

  ARIA_PROP_PREFAB_STD_STRING(public, public, , std::string, name1);

public:
  [[nodiscard]] std::vector<int> oneTwoThree() const { return {1, 2, 3}; }

  void dummy0() {}

  void dummy0() const { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  void dummy1() { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  void dummy1() const {}

  void dummy2(int v) { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  void dummy2(int v) const {}

private:
  std::string name_ = "Lua です喵"; // Test UTF-8.

  std::string ARIA_PROP_IMPL(name1)() const { return name_; }

  std::string ARIA_PROP_IMPL(name1)(const std::string &name) { name_ = name; }
};

//
//
//
#define __ARIA_LUA_NEW_USER_TYPE_BEGIN(LUA, TYPE) LUA.new_usertype<TYPE>(#TYPE

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS2(TYPE, NAME)                                                            \
  , #NAME, static_cast<decltype(std::declval<TYPE>().NAME()) (TYPE::*)()>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS3(TYPE, NAME, SPECIFIERS)                                                \
  , #NAME, static_cast<decltype(std::declval<TYPE>().NAME()) (TYPE::*)() SPECIFIERS>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD(...)                                                                           \
  __ARIA_EXPAND(                                                                                                       \
      __ARIA_EXPAND(ARIA_CONCAT(__ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS, ARIA_NUM_OF(__VA_ARGS__)))(__VA_ARGS__))

#define __ARIA_LUA_NEW_USER_TYPE_END )

//
//
//
#define ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, type) __ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, type)

#define ARIA_LUA_NEW_USER_TYPE_METHOD(...) __ARIA_EXPAND(__ARIA_LUA_NEW_USER_TYPE_METHOD(__VA_ARGS__))

#define ARIA_LUA_NEW_USER_TYPE_END __ARIA_LUA_NEW_USER_TYPE_END

} // namespace

//
//
//
//
//
TEST(Lua, OwnershipAndInheritance) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  lua.new_usertype<GrandParent>("GrandParent", "value", &GrandParent::value);
  lua.new_usertype<Parent>("Parent", "value", &Parent::value, sol::base_classes, sol::bases<GrandParent>());
  lua.new_usertype<Child>("Child", "value", &Child::value, sol::base_classes, sol::bases<Parent>());

  std::shared_ptr<GrandParent> parent = std::make_shared<Parent>();
  lua["parent"] = parent;

  std::unique_ptr<GrandParent> child = std::make_unique<Child>();
  lua["child"] = child.get();

  std::unique_ptr<GrandParent> owningChild = std::make_unique<Child>();
  lua["owningChild"] = std::move(owningChild);

  lua.script("assert(parent:value() == 1)");
  lua.script("assert(child:value() == 2, \"Should equals to 2\")"
             "assert(owningChild:value() == 2, \"Should equals to 2\")");
  lua.script("assert(parent:value() == 1)"
             "assert(child:value() == 2, \"Should equals to 2\")"
             "assert(owningChild:value() == 2, \"Should equals to 2\")");
}

TEST(Lua, Property) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, Object)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, name0, const)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, name1)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, oneTwoThree, const)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, dummy0)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, dummy1, const)
  ARIA_LUA_NEW_USER_TYPE_END;

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, decltype(std::declval<Object>().name1()))
  ARIA_LUA_NEW_USER_TYPE_METHOD(decltype(std::declval<Object>().name1()), value)
  ARIA_LUA_NEW_USER_TYPE_END;

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
             "assert(obj:oneTwoThree()[3] == 3)"
             "obj:dummy0()"
             "obj:dummy1()"
             "obj:dummy2(1)");
}

} // namespace ARIA
