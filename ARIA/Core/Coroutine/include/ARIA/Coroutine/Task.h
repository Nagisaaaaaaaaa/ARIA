#pragma once

/// \file
/// \brief A `task` represents an asynchronous computation that is executed lazily
/// in that the execution of the coroutine does not start until the `task` is `await`ed.
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

#include <cppcoro/task.hpp>

namespace ARIA::Coroutine {

/// \brief A `task` represents an asynchronous computation that is executed lazily
/// in that the execution of the coroutine does not start until the `task` is `await`ed.
using cppcoro::task;

} // namespace ARIA::Coroutine
