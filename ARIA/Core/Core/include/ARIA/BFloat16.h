#pragma once

/// \file
/// \brief A 16-bit brain floating point (BF16) implementation:
/// 1 sign bit, 8 exponent bits, and 7 mantissa bits.
/// See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format.
///
/// `cmath` and `limits` are extended to support `bfloat16`.
//
//
//
//
//
#include "ARIA/Python.h"

#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda_bf16.h>

namespace ARIA {

/// \brief A 16-bit brain floating point (BF16) implementation:
/// 1 sign bit, 8 exponent bits, and 7 mantissa bits.
///
/// ARIA directly uses CUDA's implementation,
/// see https://docs.nvidia.com/cuda/cuda-math-api/struct____nv__bfloat16.html.
///
/// \note `cmath` and `limits` are extended to support `bfloat16`, so
/// you are able to use functions such as `std::abs`, `std::max`, and
/// `std::numeric_limits`, or the `cuda::std::` ones with `bfloat16`.
///
/// \example ```cpp
/// bfloat16 a{0.1F};
/// bfloat16 b{0.2F};
/// bfloat16 c = a + b;
///
/// bfloat16 inf0 = std::numeric_limits<bfloat16>::infinity();
/// bfloat16 inf1 = cuda::std::numeric_limits<bfloat16>::infinity();
/// ```
using bfloat16 = nv_bfloat16;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/BFloat16.inc"
