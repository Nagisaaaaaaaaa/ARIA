#pragma once

/// \brief `small_vector` is a 'vector' (really, a variable-sized array),
/// optimized for the case when the array is small.
///
/// It contains some number of elements in-place, which allows it to
/// avoid heap allocation when the actual number of elements is below that threshold.
/// This allows normal "small" cases to be fast without losing generality for large inputs.
//
//
//
//
//
#include "ARIA/ARIA.h"

#include <gch/small_vector.hpp>

namespace ARIA {

/// \brief `small_vector` is a 'vector' (really, a variable-sized array),
/// optimized for the case when the array is small.
///
/// It contains some number of elements in-place, which allows it to
/// avoid heap allocation when the actual number of elements is below that threshold.
/// This allows normal "small" cases to be fast without losing generality for large inputs.
///
/// ARIA directly uses `gch::small_vector`, see https://github.com/gharveymn/small_vector.
using gch::small_vector;

} // namespace ARIA
