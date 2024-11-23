#pragma once

#include "ARIA/Auto.h"

namespace ARIA {

#define let ::ARIA::property::detail::NonProxyType auto

template <ARIA::property::detail::ProxyType T>
ARIA_HOST_DEVICE auto Let(T &&v) {
  return Auto(std::forward<T>(v));
}

} // namespace ARIA
