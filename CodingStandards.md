# Coding Standards

[TOC]

## Mechanical Source Issues

1. Use `#pragma once` instead of header guard.

2. Prefer C++-style casts.

   When casting, use `static_cast`, `reinterpret_cast`, and `const_cast`, rather than C-style casts. There are two
   exceptions to this:

    1. When casting to `void` to suppress warnings about unused variables (as an alternative to `[[maybe_unused]]`).
       Prefer C-style casts in this instance.
    2. When casting between integral types (including enums that are not strongly-typed), functional-style casts are
       permitted as an alternative to `static_cast`.

3. Make anonymous `namespace`s as small as possible, and only use them for class declarations.

   The problem with anonymous `namespace`s is that they naturally want to encourage indentation of their body, and they
   reduce locality of reference: if you see a random function definition in a C++ file, it is easy to see if it is
   marked `static`, but seeing if it is in an anonymous `namespace` requires scanning a big chunk of the file.

   Because of this, we have a simple guideline: make anonymous `namespace`s as small as possible, and only use them for
   class declarations. For example:

   ```c++
   namespace {
   class StringSort final {
   ...
   public:
     StringSort(...)
     bool operator<(const char *rhs) const;
   };
   } // namespace
   
   static void runHelper() {
     ...
   }
   
   bool StringSort::operator<(const char *rhs) const {
     ...
   }
   ```

4. Use `constinit` instead of `static`.

   Globals in different source files are initialized in arbitrary order, making the code more difficult to reason about.
   Use `constinit` instead of `static`.

   ```c++
   // Yes
   extern constinit int staticA;
   
   // No
   extern int staticA;
   ```

5. Use of class and struct Keywords.

    1. All declarations and definitions of a given `class` or `struct` must use the same keyword. For example:

       ```c++
       struct Example;
       
       // Yes
       struct Example { ... };
       
       // No
       class Example { ... };
       ```

    2. When `class` is used, *all* member variables are declared `private`.

    3. When `struct` is used, *all* member variables are declared `public`.

6. Whenever possible, use `{}` to call a constructor instead of `()`.

   `{}` prevents narrowing conversions, and it is immune to C++'s most vexing parse.

   ```c++
   // Yes
   SomeType a{...};
   
   // No
   SomeType a(...);
   ```

7. Always use `auto + Auto()` type deduction to make the code safe.

    1. You must assume that proxies such as `std::vector<bool>` are used everywhere.

    2. Never use `auto` without `Auto()` to prevent undefined behavior caused by proxies.

    3. Use `auto + Auto()` if and only if it makes the code more readable or easier to maintain.

       ```c++
       auto b = Auto(a);
       ```

8. Beware unnecessary copies with `auto`.

   ```c++
   for (const auto& v : values) { ... }
   ```

## Style Issues

1. `#include` as little as possible.

2. Use early exists and `continue` to simplify code.

3. Name properly.

   ```c++
   namespace ARIA {
   
   class Transform final {
   public:
     [[nodiscard]] Vec3r position() const noexcept {
       return position_;
     }
     
     void Rotate(...) {
       int a = 1;
       const int b = 1;
       constexpr int C = 1;
       
       ...
     }
     
   private:
     Vec3r position_;
   }
   
   
   
   enum class Flag0 : unsigned {
     A = 0,
     B,
     ...
     Count
   }
   enum class Flag1 {
     A,
     B,
     ...
   }
   enum Flag2 {
     F_A,
     F_B,
     ...
   }
   enum class Flag3 : unsigned {
     A = 1,
     B = 1 << 1;
     ...
   }
   
   } // namespace ARIA
   ```

   ```c++
   // Time.h
   namespace ARIA {
   namespace Time {
   
   [[nodiscard]] double deltaTime() noexcept;
   
   } // namespace Time
   } // namespace ARIA
   
   
   
   // Time.cpp
   namespace ARIA {
   namespace Time {
   
   constinit double deltaTime_ = 0;
   
   double deltaTime() {
     return time;
   }
   
   } // namespace Time
   } // namespace ARIA
   ```

4. Class design.

    1. Make all classes be one of two kinds.
        1. Abstract base classes that are never instantiated.
        2. Concrete classes that are derived from one of these base classes.

    2. Override and final.

       ```c++
       class A {
       private:
         virtual void Func() = 0;
       }
       class B : public A {
       private:
         void Func() override { ... }
       }
       class C final : public B {
       private:
         void Func() final { ... }
       }
       ```

    3. Use `namespace`s instead of `static` classes

5. Assert liberally.

   ```
   assert(!v.empty() && "Vector should not be empty");
   assert(false && "Vector should not be empty");
   ```

6. Use range-based `for` loops whenever possible.

7. A member function defined in a class definition is implicitly inline, so don't put the `inline` keyword in this case.

   ```c++
   class Foo {
   public:
     // Yes
     void bar() {
       // ...
     }
     
     // No
     inline void bar() {
       // ...
     }
   };
   ```

8. Use `++i` instead of `i++`.

9. Don't use `{}` on simple single-statement bodies of `if`/`else`/loop statements.

