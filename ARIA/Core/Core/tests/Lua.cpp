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

  std::vector<int> dummy3(std::string v0, const std::vector<int> &v1) { return v1; }

  std::vector<int> dummy3(std::string v0, const std::vector<int> &v1) const {
    ARIA_THROW(std::runtime_error, "This method should never be called");
    return v1;
  }

private:
  std::string name_ = "Lua です喵"; // Test UTF-8.

  std::string ARIA_PROP_IMPL(name1)() const { return name_; }

  std::string ARIA_PROP_IMPL(name1)(const std::string &name) { name_ = name; }
};

//
//
//
#define __ARIA_LUA_NEW_USER_TYPE_BEGIN(LUA, TYPE) LUA.new_usertype<TYPE>(#TYPE

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS3(CONST_OR_EMPTY, TYPE, NAME)                                            \
  , #NAME, static_cast<decltype(std::declval<TYPE>().NAME()) (TYPE::*)() CONST_OR_EMPTY>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS4(CONST_OR_EMPTY, TYPE, NAME, T0)                                        \
  , #NAME, static_cast<decltype(std::declval<TYPE>().NAME(std::declval<T0>())) (TYPE::*)(T0)CONST_OR_EMPTY>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS5(CONST_OR_EMPTY, TYPE, NAME, T0, T1)                                    \
  , #NAME,                                                                                                             \
      static_cast<decltype(std::declval<TYPE>().NAME(std::declval<T0>(), std::declval<T1>())) (TYPE::*)(               \
          T0, T1)CONST_OR_EMPTY>(&TYPE::NAME)

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

TEST(Lua, MethodsAndProperties) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  static_assert(std::is_same_v<int &, decltype(std::declval<int &>())>);

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, Object)
  ARIA_LUA_NEW_USER_TYPE_METHOD(const, Object, name0)
  ARIA_LUA_NEW_USER_TYPE_METHOD(, Object, name1)
  ARIA_LUA_NEW_USER_TYPE_METHOD(const, Object, oneTwoThree)
  ARIA_LUA_NEW_USER_TYPE_METHOD(, Object, dummy0)
  ARIA_LUA_NEW_USER_TYPE_METHOD(const, Object, dummy1)
  ARIA_LUA_NEW_USER_TYPE_METHOD(const, Object, dummy2, int)
  ARIA_LUA_NEW_USER_TYPE_METHOD(, Object, dummy3, std::string, const std::vector<int> &)
  ARIA_LUA_NEW_USER_TYPE_END;

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, decltype(std::declval<Object>().name1()))
  ARIA_LUA_NEW_USER_TYPE_METHOD(, decltype(std::declval<Object>().name1()), value)
  ARIA_LUA_NEW_USER_TYPE_END;

  Object obj;
  lua["obj"] = &obj;

  lua.script("local name0 = obj:name0()\n"
             "local name1 = obj:name1()\n"
             "\n"
             "assert(name0 == \"Lua です喵\", \"Error message です喵\")\n"
             "assert(tostring(name1) == \"Lua です喵\", \"Error message です喵\")\n"
             "assert(name1:value() == \"Lua です喵\", \"Error message です喵\")\n"
             "assert(obj:oneTwoThree()[1] == 1)\n"
             "assert(obj:oneTwoThree()[2] == 2)\n"
             "assert(obj:oneTwoThree()[3] == 3)\n"
             "obj:dummy0()\n"
             "obj:dummy1()\n"
             "obj:dummy2(1)\n"
             "obj:dummy3(obj:name0(), obj:oneTwoThree())\n"
             "obj:dummy3(\"Lua です喵\", obj:oneTwoThree())\n");
}

} // namespace ARIA
