# ARIA

ARIA is a collection of foundational computer graphics infrastructure **for research**.

ARIA is a melting pot where you can find many interesting things, such as:

1. `Property`: C#-like properties.
   1. Define a C#-like property with several lines of codes.
   2. Even stronger than the C# built-in features.
2. `Array`, `Vector`: Policy-based arrays and vectors.
   1. Support CPU or GPU storages.
   2. Support automatic AoS to SoA layouts.
3. `TensorVector`: Policy-based multidimensional arrays and views.
   1. Support CPU or GPU storages.
   2. Support arbitrary multidimensional layouts.
   3. Support fully or partially compile-time-determined layouts.
   4. Support automatic AoS to SoA layouts.
4. `VDB`: Light-weighted and policy-based `VDB`.
   1. Much slower than `OpenVDB` and `NanoVDB`.
   2. Light-weighted and easy to compile.
   3. Support GPU storages.
   4. Support thread-safe memory allocations.
   5. Support thread-safe read and write accesses.
   6. Support kernel launches for each valid coordinate.
   7. Support arbitrary large and small coordinates.
   8. Support automatic AoS to SoA layouts.
5. `Object`, `Component`, `Transform`: Unity-like hierarchical objects.
   1. Powered with C#-like `Property`s.
   2. Interfaces are almost the same as Unity.
6. Many other interesting features, see the documents in headers.

**ARIA is extremely radical:**

1. Compiler support: Fully cross-platform but at least C++ 20 and CUDA 12.
2. Usually, we have to modify several lines of codes before compiling...
3. Interfaces may be revised without any warnings.

ARIA adheres to strict coding standards, and we add as many comments as possible to each file, including the usage of interfaces and implementation details, because we do not assume that every graphics researcher is highly familiar with C++. Even if you have only a rudimentary understanding of modern C++, you can use ARIA with confidence.

Here lists all robust modules:

- `ARIA::Core::Concurrency`,
- `ARIA::Core::Core`,
- `ARIA::Core::Coroutine`,
- `ARIA::Core::Geometry`,
- `ARIA::Core::Math`,
- `ARIA::Scene::Scene`.

## Getting Started

This tutorial shows how to integrate ARIA into a simple project with cmake and CPM.

1. Download the latest CUDA, see https://developer.nvidia.com/cuda-downloads.

   (Currently, ARIA cannot compile without CUDA. We will fix it in the future.)

2. Suppose your project name is `ProjName`. create the following directories and files:

   ```bash
   ProjName/
   ├─ cmake/
   ├─ CMakeLists.txt
   ├─ main.cpp
   ```

3. Copy `CPM.cmake` to `cmake/`, see https://github.com/cpm-cmake/CPM.cmake.

   ```bash
   ProjName/
   ├─ cmake/
   │  ├─ CPM.cmake
   ├─ CMakeLists.txt
   ├─ main.cpp
   ```

4. Edit `CMakeLists.txt`:

   ```cmake
   cmake_minimum_required(VERSION 3.25.2)
   project(ProjName LANGUAGES CXX)
   
   list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_BINARY_DIR}")
   
   include(CPM)
   
   CPMAddPackage(
       NAME ARIA
       GITHUB_REPOSITORY Nagisaaaaaaaaa/ARIA
       GIT_TAG main
       OPTIONS "ARIA_BUILD_TESTS OFF"
   )
   
   add_executable(${PROJECT_NAME} main.cpp)
   
   target_link_libraries(${PROJECT_NAME} PUBLIC
       ARIA::Core::Core
   )
   ```

5. Reload cmake.

6. Edit `main.cpp`:

   ```c++
   #include <ARIA/ForEach.h>
   
   int main() {
     std::string s = "Hello ARIA!";
   
     ARIA::ForEach(s.length(), [&](auto i) {
       fmt::print("{}", s[i]);
     });
     fmt::print("\n");
   
     return 0;
   }
   ```

7. Now, we are ready to compile the codes.

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

4. **Even stronger than C#.**

   The ARIA property system is even stronger than the C# built-in features. You can write properties with arbitrary number and type of parameters. For example, suppose you are writing a 2D fluid simulator based on the lattice Boltzmann method (LBM), your code may look like this:

   ```c++
   using I0 = std::integral_constant<int, 0>;
   using I1 = std::integral_constant<int, 1>;
   using I2 = std::integral_constant<int, 2>;
   ...

   // The streaming process of the LBM.
   grid.f(coord, I0{}) = grid.fPost(coord - Coord{0, 0}, I0{});
   grid.f(coord, I1{}) = grid.fPost(coord - Coord{1, 0}, I1{});
   grid.f(coord, I2{}) = grid.fPost(coord - Coord{0, 1}, I2{});
   ...
   ```

   Here, both properties, `f` and `fPost`, have 2 parameters, where the first type is `Coord` (means coordinate), and the second type can be any `std::integral_constant` (means the LBM velocity set).

   Only several lines of codes are needed to generate such complex properties.

5. **Make `auto` safe in ARIA.**

   To make `auto` safe, ARIA uses `auto + Auto()` type deduction:
   
   ```c++
   Vec3r a = {1, 2, 3};
   Vec3r b = {2, 4, 6};
   
   // NO.
   auto c = a.cross(b);
   
   // YES.
   Vec3r c0 = a.cross(b);
   auto c1 = Auto(a.cross(b)); // With `auto + Auto()` type deduction, type of `c1` is correctly deduced to `Vec3r`.
   ```
   
   `Auto()` helps better deduce the types from all proxy systems used in ARIA, which makes our codes safe. Read the comments in `Auto.h` and `Property.h` to see how to use them.

6. **Make life easier for small projects.**

   Suppose you are writing a very small project based on ARIA, it is very annoying to use `auto + Auto` type deduction everywhere. Instead, we want to use `Auto` only when we have to, which means that the compiler should be able to tell us:
   1. Which `auto` is unsafe and refuse to compile them,
   2. Which `Auto` is unnecessary and refuse to compile them.

   So, `let + Let` type deduction is introduced to make life easier for small projects:

   ```c++
   let x = 10;
   let x = Let(10); // Compile error.
   
   std::vector<bool> v(1);
   let y = v[0]; // Compile error.
   let y = Let(v[0]);
   ```

   Read the comments in `Let.h` to see how to use them, and feel free to use `let + Let` instead of `auto + Auto`.

7. **Coding standards.**

   ARIA uses the coding standards similar to https://llvm.org/docs/CodingStandards.html but very different in naming. Please exactly follow the style of `ARIA::Core`. Read the codes and you will understand.

