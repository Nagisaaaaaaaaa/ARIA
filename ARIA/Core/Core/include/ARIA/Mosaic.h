#pragma once

// TODO: Mosaic is an abstraction about:
//       How to describe instances of one type with instances of another type.
//       Eg 1. `double` can be described with `float` with precision lost.
//       Eg 2. `double` can be described with `double` itself.
//       Eg 3. `Vec3f` can be described with `struct { float, float, float }`.

// TODO: Document that `boost::pfr` fails to handle:
//       1. Classes with only one member.
//       2. Inheritance.
//       3. All non-scalar and non-aggregate classes, for example,
//          `boost::pfr::get<0>(std::string{})` will fails to compile.

#include "ARIA/TypeArray.h"

#include <boost/pfr.hpp>

namespace ARIA {

template <typename T, typename TMosaicPattern>
class Mosaic;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/Mosaic.inc"
