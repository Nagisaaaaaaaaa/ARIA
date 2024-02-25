#include "ARIA/detail/Macros.h"

#include <gtest/gtest.h>

#include <array>

namespace ARIA {

namespace {

#define SQUARE(x) ((x) * (x))

#define ZERO  0
#define ONE   1
#define FALSE false
#define TRUE  true

class A {
public:
  ARIA_COPY_ABILITY(A, delete);
  ARIA_MOVE_ABILITY(A, default);
};

class B {
public:
  ARIA_COPY_MOVE_ABILITY(B, delete, default);
};

} // namespace

TEST(Macros, Base) {
  // Number of.
  EXPECT_EQ(ARIA_NUM_OF(1, 2), 2);
  EXPECT_EQ(ARIA_NUM_OF(1, 2, 3), 3);

  EXPECT_EQ(ARIA_NUM_OF(a, b), 2);
  EXPECT_EQ(ARIA_NUM_OF(a, b, c), 3);

  // Concatenate.
#if 0 // TODO: clang fails to compile the following codes.
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (-5)), 25);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (-3)), 9);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (-2)), 4);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (-1)), 1);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (1)), 1);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (2)), 4);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (3)), 9);
  EXPECT_EQ(ARIA_CONCAT(SQUARE, (5)), 25);
#endif

  // Conditional.
  EXPECT_EQ(ARIA_COND(0, 5, 6), 6);
  EXPECT_EQ(ARIA_COND(1, 5, 6), 5);

  EXPECT_EQ(ARIA_COND(ZERO, 5, 6), 6);
  EXPECT_EQ(ARIA_COND(ONE, 5, 6), 5);

  EXPECT_EQ(ARIA_COND(false, 5, 6), 6);
  EXPECT_EQ(ARIA_COND(true, 5, 6), 5);

  EXPECT_EQ(ARIA_COND(FALSE, 5, 6), 6);
  EXPECT_EQ(ARIA_COND(TRUE, 5, 6), 5);

  int five = ARIA_COND(ARIA_IS_HOST_CODE, 5, 6);
  EXPECT_EQ(five, 5);

  {
    bool failed = false;
    try {
      ARIA_IF(true, throw std::exception{};);
    } catch (...) { failed = true; }

    EXPECT_TRUE(failed);
  }

  {
    bool failed = false;
    try {
      ARIA_IF(false, throw std::exception{};);
    } catch (...) { failed = true; }

    EXPECT_FALSE(failed);
  }

  // Throw.
  {
    bool failed = false;
    try {
      ARIA_THROW(std::runtime_error, "Runtime error");
    } catch (...) { failed = true; }

    EXPECT_TRUE(failed);
  }

  // Assertions and exceptions.
  // ARIA_UNIMPLEMENTED;
  // ARIA_ASSERT(false);
  // ARIA_ASSERT(false, "Message");
}

} // namespace ARIA
