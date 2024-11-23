#pragma once

/// \file Make sure you are familiar with `Auto.h` before continue.
///
/// Suppose you are writing a very small project based on ARIA,
/// it is very annoying to use `auto + Auto` type deduction everywhere.
/// Instead, we want to use `Auto` only when we have to, which
/// means that the compiler should be able to tell us:
///   1. Which `auto` is unsafe and refuse to compile them,
///   2. Which `Auto` is unnecessary and refuse to compile them.
///
/// It should be warned that such a design is bad for large projects, because
/// we usually need template functions which can
/// handle both proxy types and non-proxy types.
/// That is where `auto + Auto` type deduction is necessary.
///
/// So, if you are writing a very small project, feel free to
/// include this file and use `let + Let` instead of `auto + Auto`.

//
//
//
//
//
#include "ARIA/Auto.h"

namespace ARIA {

#define let ::ARIA::property::detail::NonProxyType auto

template <ARIA::property::detail::ProxyType T>
ARIA_HOST_DEVICE auto Let(T &&v) {
  return Auto(std::forward<T>(v));
}

} // namespace ARIA
