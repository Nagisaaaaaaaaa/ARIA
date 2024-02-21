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
  // Is invocable with brackets.
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

  // Is invocable with brackets (r).
  static_assert(std::is_invocable_r_v<int, CallByParentheses, int>);
  static_assert(std::is_invocable_r_v<float, CallByParentheses, int>);
  static_assert(std::is_invocable_r_v<uint64, CallByParentheses, int>);

  static_assert(is_invocable_with_brackets_r_v<int, std::vector<int>, int>);
  static_assert(is_invocable_with_brackets_r_v<double, std::vector<int>, uint64>);
  static_assert(is_invocable_with_brackets_r_v<uint64, std::vector<int>, double>);

  static_assert(is_invocable_with_brackets_r_v<int, CallByBrackets, int>);
  static_assert(is_invocable_with_brackets_r_v<double, CallByBrackets, uint64>);
  static_assert(is_invocable_with_brackets_r_v<uint64, CallByBrackets, double>);

  static_assert(is_invocable_with_brackets_r_v<int, CallByBoth, int>);
  static_assert(is_invocable_with_brackets_r_v<double, CallByBoth, uint64>);
  static_assert(is_invocable_with_brackets_r_v<uint64, CallByBoth, double>);

  static_assert(!is_invocable_with_brackets_r_v<int, CallByBrackets, int, int>);
  static_assert(!is_invocable_with_brackets_r_v<int, CallByParentheses, int>);
  static_assert(!is_invocable_with_brackets_r_v<std::string, CallByBrackets, int>);
}

} // namespace ARIA
