#include "ARIA/Life/Offline/MonoBehavior.h"

namespace ARIA {

//! `Registry` related functions should never be called `inline`, to
//! make it consistent with `static` variables of `Registry`.
OMonoBehavior::Range<OMonoBehavior::Iterator<OMonoBehavior>, OMonoBehavior::Iterator<OMonoBehavior>>
OMonoBehavior::range() noexcept {
  return Base::range();
}

OMonoBehavior::OMonoBehavior() = default;

OMonoBehavior::~OMonoBehavior() noexcept = default;

} // namespace ARIA
