#include "ARIA/Invocations.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class Float {
public:
  explicit operator float() const { return 0.0F; }
};

class CallByParentheses {
public:
  Float operator()(int v) {}
};

class CallByBrackets {
public:
  Float operator[](int v) {}
};

class CallByBoth {
public:
  Float operator()(int v) {}

  Float operator[](int v) {}
};

} // namespace

TEST(Concepts, Base) {
  // Is invocable with brackets.
  static_assert(is_invocable_with_brackets_v<std::vector<std::string>, int>);
  static_assert(is_invocable_with_brackets_v<std::vector<std::string>, uint64>);
  static_assert(is_invocable_with_brackets_v<std::vector<std::string>, double>);

  static_assert(is_invocable_with_brackets_v<CallByBrackets, int>);
  static_assert(is_invocable_with_brackets_v<CallByBrackets, uint64>);
  static_assert(is_invocable_with_brackets_v<CallByBrackets, double>);

  static_assert(is_invocable_with_brackets_v<CallByBoth, int>);
  static_assert(is_invocable_with_brackets_v<CallByBoth, uint64>);
  static_assert(is_invocable_with_brackets_v<CallByBoth, double>);

  static_assert(!is_invocable_with_brackets_v<CallByBrackets, int, int>);
  static_assert(!is_invocable_with_brackets_v<CallByParentheses, int>);

  // Invocable with brackets.
  static_assert(invocable_with_brackets<std::vector<std::string>, int>);
  static_assert(invocable_with_brackets<std::vector<std::string>, uint64>);
  static_assert(invocable_with_brackets<std::vector<std::string>, double>);

  static_assert(invocable_with_brackets<CallByBrackets, int>);
  static_assert(invocable_with_brackets<CallByBrackets, uint64>);
  static_assert(invocable_with_brackets<CallByBrackets, double>);

  static_assert(invocable_with_brackets<CallByBoth, int>);
  static_assert(invocable_with_brackets<CallByBoth, uint64>);
  static_assert(invocable_with_brackets<CallByBoth, double>);

  static_assert(!invocable_with_brackets<CallByBrackets, int, int>);
  static_assert(!invocable_with_brackets<CallByParentheses, int>);

  // Is invocable with brackets (r).
  static_assert(is_invocable_with_brackets_r_v<std::string, std::vector<std::string>, int>);
  static_assert(is_invocable_with_brackets_r_v<int, std::vector<float>, int>);
  static_assert(is_invocable_with_brackets_r_v<double, std::vector<float>, uint64>);
  static_assert(is_invocable_with_brackets_r_v<uint64, std::vector<float>, double>);

  static_assert(std::is_invocable_r_v<Float, CallByParentheses, int>);
  static_assert(!std::is_invocable_r_v<float, CallByParentheses, int>);

  static_assert(is_invocable_with_brackets_r_v<Float, CallByBrackets, int>);
  static_assert(!is_invocable_with_brackets_r_v<float, CallByBrackets, double>);
  static_assert(!is_invocable_with_brackets_r_v<std::string, CallByBrackets, int>);

  static_assert(is_invocable_with_brackets_r_v<Float, CallByBoth, int>);
  static_assert(!is_invocable_with_brackets_r_v<float, CallByBoth, double>);
  static_assert(!is_invocable_with_brackets_r_v<std::string, CallByBoth, double>);

  static_assert(!is_invocable_with_brackets_r_v<int, CallByBrackets, int, int>);
  static_assert(!is_invocable_with_brackets_r_v<int, CallByParentheses, int>);
}

} // namespace ARIA
