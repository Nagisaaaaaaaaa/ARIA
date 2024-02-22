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

class CallByBothVoid {
public:
  void operator[](int v) {}
};

} // namespace

TEST(Invocations, Base) {
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

  static_assert(is_invocable_with_brackets_v<CallByBothVoid, int>);
  static_assert(is_invocable_with_brackets_v<CallByBothVoid, uint64>);
  static_assert(is_invocable_with_brackets_v<CallByBothVoid, double>);

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

  static_assert(invocable_with_brackets<CallByBothVoid, int>);
  static_assert(invocable_with_brackets<CallByBothVoid, uint64>);
  static_assert(invocable_with_brackets<CallByBothVoid, double>);

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

  static_assert(std::is_invocable_r_v<void, CallByBoth, int>);

  static_assert(is_invocable_with_brackets_r_v<void, CallByBothVoid, int>);

  static_assert(!is_invocable_with_brackets_r_v<int, CallByBrackets, int, int>);
  static_assert(!is_invocable_with_brackets_r_v<int, CallByParentheses, int>);
}

TEST(Invocations, Invoke) {
  // Invoke with brackets.
  {
    static_assert(std::is_same_v<decltype(invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0)), float &>);
    decltype(auto) v = invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0);
    static_assert(std::is_same_v<decltype(v), float &>);

    EXPECT_FLOAT_EQ(invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0), 1.1F);
    EXPECT_FLOAT_EQ(invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 1), 2.2F);
    EXPECT_FLOAT_EQ(invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 2), 3.3F);
  }

  // Invoke with parentheses or brackets.
  {
    {
      static_assert(
          std::is_same_v<decltype(invoke_with_parentheses_or_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0)),
                         float &>);
      decltype(auto) v = invoke_with_parentheses_or_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0);
      static_assert(std::is_same_v<decltype(v), float &>);

      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0), 1.1F);
      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 1), 2.2F);
      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 2), 3.3F);
    }

    {
      std::vector<float> storage = {1.1F, 2.2F, 3.3F};
      auto vs = [&](size_t i) -> decltype(auto) { return storage[i]; };

      static_assert(std::is_same_v<decltype(invoke_with_parentheses_or_brackets(vs, 0)), float &>);
      decltype(auto) v = invoke_with_parentheses_or_brackets(vs, 0);
      static_assert(std::is_same_v<decltype(v), float &>);

      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(vs, 0), 1.1F);
      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(vs, 1), 2.2F);
      EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(vs, 2), 3.3F);
    }
  }

  // Invoke with brackets or parentheses.
  {
    {
      static_assert(
          std::is_same_v<decltype(invoke_with_brackets_or_parentheses(std::vector<float>{1.1F, 2.2F, 3.3F}, 0)),
                         float &>);
      decltype(auto) v = invoke_with_brackets_or_parentheses(std::vector<float>{1.1F, 2.2F, 3.3F}, 0);
      static_assert(std::is_same_v<decltype(v), float &>);

      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(std::vector<float>{1.1F, 2.2F, 3.3F}, 0), 1.1F);
      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(std::vector<float>{1.1F, 2.2F, 3.3F}, 1), 2.2F);
      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(std::vector<float>{1.1F, 2.2F, 3.3F}, 2), 3.3F);
    }

    {
      std::vector<float> storage = {1.1F, 2.2F, 3.3F};
      auto vs = [&](size_t i) -> decltype(auto) { return storage[i]; };

      static_assert(std::is_same_v<decltype(invoke_with_brackets_or_parentheses(vs, 0)), float &>);
      decltype(auto) v = invoke_with_brackets_or_parentheses(vs, 0);
      static_assert(std::is_same_v<decltype(v), float &>);

      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(vs, 0), 1.1F);
      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(vs, 1), 2.2F);
      EXPECT_FLOAT_EQ(invoke_with_brackets_or_parentheses(vs, 2), 3.3F);
    }
  }
}

} // namespace ARIA
