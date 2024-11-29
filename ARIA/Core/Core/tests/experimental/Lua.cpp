#include "ARIA/Property.h"
#include "ARIA/PropertySTL.h"

#define SOL_ALL_SAFETIES_ON 1

#include <gtest/gtest.h>
#include <sol/sol.hpp>

namespace ARIA {

namespace {

#define __ARIA_LUA_NEW_USER_TYPE_BEGIN(LUA, TYPE) LUA.new_usertype<TYPE>(#TYPE

// clang-format off
#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS3(TYPE, SPECIFIERS, NAME)                                                \
  , #NAME, static_cast<decltype(std::declval<SPECIFIERS TYPE>().NAME(                                                  \
  ))                                                                                                                   \
  (TYPE::*)() SPECIFIERS>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS4(TYPE, SPECIFIERS, NAME, T0)                                            \
  , #NAME, static_cast<decltype(std::declval<SPECIFIERS TYPE>().NAME(                                                  \
  std::declval<T0>()))                                                                                                 \
  (TYPE::*)(T0) SPECIFIERS>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS5(TYPE, SPECIFIERS, NAME, T0, T1)                                        \
  , #NAME, static_cast<decltype(std::declval<SPECIFIERS TYPE>().NAME(                                                  \
  std::declval<T0>(), std::declval<T1>()))                                                                             \
  (TYPE::*)(T0, T1) SPECIFIERS>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS6(TYPE, SPECIFIERS, NAME, T0, T1, T2)                                    \
  , #NAME, static_cast<decltype(std::declval<SPECIFIERS TYPE>().NAME(                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>()))                                                         \
  (TYPE::*)(T0, T1, T2) SPECIFIERS>(&TYPE::NAME)

#define __ARIA_LUA_NEW_USER_TYPE_METHOD_PARAMS7(TYPE, SPECIFIERS, NAME, T0, T1, T2, T3)                                \
  , #NAME, static_cast<decltype(std::declval<SPECIFIERS TYPE>().NAME(                                                  \
  std::declval<T0>(), std::declval<T1>(), std::declval<T2>(), std::declval<T3>()))                                     \
  (TYPE::*)(T0, T1, T2, T3) SPECIFIERS>(&TYPE::NAME)
// clang-format on

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

//
//
//
//
//
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
class ConstOrNonConst {
public:
  std::vector<int> oneTwoThree() const { return {1, 2, 3}; }

  void func0() {}

  void func0() const { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  int func1() { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  void func1() const {}

  void func2(int v) { ARIA_THROW(std::runtime_error, "This method should never be called"); }

  void func2(int v) const {}

  std::vector<int> func3(std::string v0, const std::vector<int> &v1) { return v1; }

  std::vector<int> func3(std::string v0, const std::vector<int> &v1) const {
    ARIA_THROW(std::runtime_error, "This method should never be called");
    return v1;
  }
};

//
//
//
class Object {
public:
  const std::string &name0() const { return name_; }

  ARIA_PROP_PREFAB_STD_STRING(public, public, , std::string, name1);

private:
  std::string name_ = "Lua です喵"; // Test UTF-8.

  std::string ARIA_PROP_GETTER(name1)() const { return name_; }

  std::string ARIA_PROP_GETTER(name1)(const std::string &name) { name_ = name; }
};

//
//
//
class ManyOverloads {
public:
  using str = std::string;

  std::vector<bool> F0() const { return {}; }

  std::vector<bool> F1(int v0) const { return {}; }

  std::vector<bool> F2(int v0, str v1) const { return {}; }

  std::vector<bool> F3(int v0, str v1, int v2) const { return {}; }

  std::vector<bool> F4(int v0, str v1, int v2, str v3) const { return {}; }

  std::vector<bool> F5(int v0, str v1, int v2, str v3, int v4) const { return {}; }

  std::vector<bool> F6(int v0, str v1, int v2, str v3, int v4, str v5) const { return {}; }

  std::vector<bool> F7(int v0, str v1, int v2, str v3, int v4, str v5, int v6) const { return {}; }

  std::vector<bool> F8(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7) const { return {}; }

  std::vector<bool> F9(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8) const { return {}; }

  std::vector<bool> F10(int v0, str v1, int v2, str v3, int v4, str v5, int v6, str v7, int v8, str v9) const {
    return {};
  }
};

} // namespace

//
//
//
//
//
TEST(Lua, Inheritance) {
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

  lua.script("assert(parent:value() == 1)\n");
  lua.script("assert(child:value() == 2, \"Should equals to 2\")\n"
             "assert(owningChild:value() == 2, \"Should equals to 2\")\n");
  lua.script("assert(parent:value() == 1)\n"
             "assert(child:value() == 2, \"Should equals to 2\")\n"
             "assert(owningChild:value() == 2, \"Should equals to 2\")\n");
}

TEST(Lua, ConstOrNonConst) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  static_assert(std::is_same_v<int &, decltype(std::declval<int &>())>);

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, ConstOrNonConst)
  ARIA_LUA_NEW_USER_TYPE_METHOD(ConstOrNonConst, const, oneTwoThree)
  ARIA_LUA_NEW_USER_TYPE_METHOD(ConstOrNonConst, , func0)
  ARIA_LUA_NEW_USER_TYPE_METHOD(ConstOrNonConst, const, func1)
  ARIA_LUA_NEW_USER_TYPE_METHOD(ConstOrNonConst, const, func2, int)
  ARIA_LUA_NEW_USER_TYPE_METHOD(ConstOrNonConst, , func3, std::string, const std::vector<int> &)
  ARIA_LUA_NEW_USER_TYPE_END;

  ConstOrNonConst v;
  lua["v"] = &v;

  std::string str;
  lua["str"] = str;

  std::vector<int> vec;
  lua["vec"] = &vec;

  lua.script("assert(v:oneTwoThree()[1] == 1)\n"
             "assert(v:oneTwoThree()[2] == 2)\n"
             "assert(v:oneTwoThree()[3] == 3)\n"
             "v:func0()\n"
             "v:func1()\n"
             "v:func2(1)\n"
             "v:func3(str, vec)\n"
             "v:func3(\"Lua です喵\", vec)\n");
}

TEST(Lua, ARIAProperties) {
  sol::state lua;
  lua.open_libraries(sol::lib::base);

  static_assert(std::is_same_v<int &, decltype(std::declval<int &>())>);

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, Object)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, const, name0)
  ARIA_LUA_NEW_USER_TYPE_METHOD(Object, , name1)
  ARIA_LUA_NEW_USER_TYPE_END;

  ARIA_LUA_NEW_USER_TYPE_BEGIN(lua, decltype(std::declval<Object>().name1()))
  ARIA_LUA_NEW_USER_TYPE_METHOD(decltype(std::declval<Object>().name1()), , value)
  ARIA_LUA_NEW_USER_TYPE_END;

  Object obj;
  lua["obj"] = &obj;

  lua.script("local name0 = obj:name0()\n"
             "local name1 = obj:name1()\n"
             "\n"
             "assert(name0 == \"Lua です喵\", \"Error message です喵\")\n"
             "assert(tostring(name1) == \"Lua です喵\", \"Error message です喵\")\n"
             "assert(name1:value() == \"Lua です喵\", \"Error message です喵\")\n");

  // TODO: It is almost impossible to directly use ARIA properties.
  //       But using sol3 properties will populate the namespace.
}

TEST(Lua, MultipleArguments) {
  // TODO: Test this.
}

} // namespace ARIA
