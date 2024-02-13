#pragma once

/// \file
/// \brief This file introduces the `sync_wait()` function, which can be used to
/// synchronously wait until the specified `awaitable` completes.
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

#include <cppcoro/sync_wait.hpp>

namespace ARIA::Coroutine {

/// \brief The `sync_wait()` function can be used to
/// synchronously wait until the specified `awaitable` completes.
using cppcoro::sync_wait;

} // namespace ARIA::Coroutine
