#pragma once

/// \file
/// \brief `Mosaic` is an abstraction about how to describe one type with
/// another type, where the later type can be easily and automatically serialized.
///
/// For example, `Vec3f` might be extremely complex, but
/// `struct Pattern { float x, y, z; };` is always simple.
///
/// Suppose ARIA knows how to convert `Vec3f` to and from `Pattern`,
/// then, ARIA can automatically do a lot of things for us.
/// For example, arrays and vectors of `Vec3f` can be automatically
/// converted to structure-of-arrays (SoA) storages, etc.
///
/// Here lists all the ARIA built-in features which
/// are compatible with `Mosaic`:
/// 1. (Nothing now, we are still working on them, QAQ.)
///
/// Users only need to define some simple types and methods,
/// see `class Mosaic` below, and all things will be ready.
///
/// \details Actually, it is better to use the name `Puzzle`, but
/// the main developer of ARIA really likes "Kin-iro Mosaic".
/// That's why `Mosaic` is used.

//
//
//
//
//
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
