#include "ARIA/Concepts.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class CallByParentheses {
public:
  int operator()(int v) {}
};

class CallByBrackets {
public:
  int operator[](int v) {}
};

class CallByBoth {
public:
  int operator()(int v) {}

  int operator[](int v) {}
};

} // namespace

TEST(Concepts, Base) {
  static_assert(is_invocable_with_brackets_v<std::vector<int>, int>);
  static_assert(is_invocable_with_brackets_v<std::vector<int>, uint64>);
  static_assert(is_invocable_with_brackets_v<std::vector<int>, double>);

  static_assert(is_invocable_with_brackets_v<CallByBrackets, int>);
  static_assert(is_invocable_with_brackets_v<CallByBrackets, uint64>);
  static_assert(is_invocable_with_brackets_v<CallByBrackets, double>);

  static_assert(is_invocable_with_brackets_v<CallByBoth, int>);
  static_assert(is_invocable_with_brackets_v<CallByBoth, uint64>);
  static_assert(is_invocable_with_brackets_v<CallByBoth, double>);

  static_assert(!is_invocable_with_brackets_v<CallByBrackets, int, int>);
  static_assert(!is_invocable_with_brackets_v<CallByParentheses, int>);
}

} // namespace ARIA
