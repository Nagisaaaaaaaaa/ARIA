#pragma once

/// \file
/// \brief This file introduces functions `when_all()` and `when_all_ready()`.
/// They can be used to create a new `awaitable` that when `co_await`ed will
/// `co_await` each of the input `awaitable`s concurrently and
/// return an aggregate of their individual results.
///
/// If you are not familiar with C++20 coroutine, read the following contexts before continue:
/// 1. A good tutorial: https://www.scs.stanford.edu/~dm/blog/c++-coroutines.html.
/// 2. A good library: https://github.com/lewissbaker/cppcoro.

//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cppcoro/when_all.hpp>

namespace ARIA::Coroutine {

/// \brief The `when_all()` function can be used to
/// create a new `awaitable` that when `co_await`ed will
/// `co_await` each of the input `awaitable`s concurrently and
/// return an aggregate of their individual results.
using cppcoro::when_all;

/// \brief The `when_all_ready()` function can be used to
/// create a new `awaitable` that completes when all of the input `awaitable`s complete.
using cppcoro::when_all_ready;

} // namespace ARIA::Coroutine
