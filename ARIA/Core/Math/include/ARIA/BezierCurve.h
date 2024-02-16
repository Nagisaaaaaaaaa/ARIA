#pragma once

#include "ARIA/Constant.h"
#include "ARIA/MovingPoint.h"

namespace ARIA {

struct DegreeDynamic {};

template <uint v>
using Degree = C<v>;

//
//
//
template <typename T, auto dim, typename TDegree>
class BezierCurve;

//
//
//
template <typename T, auto dim>
class BezierCurve<T, dim, DegreeDynamic> {
public:
private:
};

//
//
//
template <typename T, auto dim, uint degree>
class BezierCurve<T, dim, Degree<degree>> {
public:
private:
};

} // namespace ARIA
