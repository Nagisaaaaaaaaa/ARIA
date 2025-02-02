#pragma once

#include "ARIA/detail/MosaicIterator.h"

namespace ARIA {

namespace mosaic::detail {

template <typename TMosaic, typename... Ts>
  requires(is_mosaic_v<TMosaic>)
class MosaicVector final {
private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;

public:
  using value_type = T;
};

} // namespace mosaic::detail

} // namespace ARIA
