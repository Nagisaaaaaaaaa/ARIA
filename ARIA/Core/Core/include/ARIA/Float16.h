#pragma once

/// \file
/// \brief A 16-bit half-precision floating point (FP16) implementation:
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
/// See https://en.wikipedia.org/wiki/Half-precision_floating-point_format.
///
/// `cmath` and `limits` are extended to support `float16`.
//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda_fp16.h>

namespace ARIA {

/// \brief A 16-bit half-precision floating point (FP16) implementation:
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
///
/// ARIA directly uses CUDA's implementation,
/// see https://docs.nvidia.com/cuda/cuda-math-api/struct____half.html.
///
/// \note `cmath` and `limits` are extended to support `float16`, so
/// you are able to use functions such as `std::abs`, `std::max`, and
/// `std::numeric_limits`, or the `cuda::std::` ones with `float16`.
///
/// \example ```cpp
/// float16 a{0.1F};
/// float16 b{0.2F};
/// float16 c = a + b;
///
/// float16 inf0 = std::numeric_limits<float16>::infinity();
/// float16 inf1 = cuda::std::numeric_limits<float16>::infinity();
/// ```
using float16 = half;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/Float16.inc"
