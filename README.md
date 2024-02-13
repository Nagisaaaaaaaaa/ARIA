# ARIA

ARIA is a collection of foundational computer graphics infrastructure **for research**.

**WARNING: ARIA is undergoing a complete rewrite, and to date, only a small portion has been rewritten. As a result, only the following modules are stable:**

- `ARIA::Core::Concurrency`,
- `ARIA::Core::Core`,
- `ARIA::Core::Coroutine`,
- `ARIA::Core::Math`.

ARIA is a melting pot where you can find a plethora of interesting things, such as a C++ implementation of C# properties, multi-dimensional arrays and views, and a hierarchical object and component system similar to Unity's, with more powerful infrastructures to be added in the future.

ARIA adheres to strict coding standards, and we add as many comments as possible to each file, including the usage of interfaces and implementation details, because we do not assume that every graphics researcher is highly familiar with C++. Even if you have only a rudimentary understanding of modern C++, you can use ARIA with confidence.

## Coding Guidelines

1. **What is a proxy?**

   You may have known that `std::vector<bool>` is a special case in C++ STL. See https://en.cppreference.com/w/cpp/container/vector_bool if you are not familiar with it. The signature of `operator[]` looks like this:

   ```c++
   reference operator[](size_type pos);
   const_reference operator[](size_type pos) const;
   ```
   Anything wrong? Consider the following example:

   ```c++
   std::vector<bool> v(1);
   auto x = v[0];
   std::cout << x << std::endl; // 0
   
   v[0] = true;
   std::cout << x << std::endl; // ?
   ```

   It will print `0` at the first time, easy. But, how about the second time? It will be `1`, not `0`! That is because `auto` was not deduced to `bool`, instead, it was deduced to "a magic reference" to `bool` (As STL says).

   In ARIA, we call this kind of reference as a *proxy*.

2. **`auto` is dangerous in ARIA.**

   Many people may have told you that you should use `auto` as much as possible. But as you have seen in the above example, `auto` does not work well with proxies. You may argue that `std::vector<bool>` is not a common case. That's right, but, ARIA uses other proxies almost everywhere. Here lists the currently used proxies:

   - `std::vector<bool>`,
   - `thrust::device_reference`,
   - `Eigen`,

   and most importantly, **ARIA has its own proxy generator system and ARIA heavily relies on it**. We call it the *property* system (because it is very similar to the C# built-in feature with the same name). See `Property.h`.

3. **Why property so important?**

   Suppose `class Transform` represents position, rotation and scale of an `Object`. All game engines implement hierarchical object systems. `Transform` of each `Object` not only contains its `localPosition`, `localRotation`, and `localScale`. It should also be able to compute for example, `position` and `rotation`, which represent position and rotation in world coordinate.

   Traditionally, if we want to get or set these things, we should declare methods such as: `GetPosition`, `SetPosition`, `GetRotation`, and `SetRotaion`. It works, but not elegant. The ARIA property system makes it able to:

   ```c++
   Object obj = ...;
   
   obj.transform().localPosition() = {1_R, 2_R, 3_R};
   obj.transform().localRotation() = {1_R, 0_R, 0_R, 0_R};
   
   // No longer need to call the redundant `SetPosition` and `SetRotation`.
   obj.transform().position() += {1_R, 2_R, 3_R};
   obj.transform().rotation() *= {1_R, 0_R, 0_R, 0_R};
   
   // We can even directly set their members, that is, property can be recursive.
   obj.transform().position().x() += 1_R;
   obj.transform().rotation().w() *= 2_R;
   ```

   We can use `position()`, `rotation`, `position().x()`, and `rotation.w()` as if these functions return references to the underlying member variables, but actually, these variables do not exist. Now, we are able to write C#-like elegant codes in C++, as if we are using Unity!

4. **Make `auto` and `decltype(auto)` safe in ARIA.**

   To make `auto` and `decltype(auto)` safe, ARIA uses `auto + Auto()` and `decltype(auto) + DecltypeAuto()` type deduction:
   
   ```c++
   Vec3r a = {1, 2, 3};
   Vec3r b = {2, 4, 6};
   
   // NO.
   auto c = a.cross(b);
   
   // YES.
   Vec3r c0 = a.cross(b);
   auto c1 = Auto(a.cross(b)); // With `auto + Auto()` type deduction, type of `c1` is correctly deduced to `Vec3r`.
   ```
   
   These two functions, `Auto()` and `DecltypeAuto()`, help better deduce the types from all proxy systems used in ARIA, which makes our codes safe. Read the comments in `Auto.h` and `Property.h` to see how to use them.


5. **See `CodingStandards.md` for the detailed coding rules.**