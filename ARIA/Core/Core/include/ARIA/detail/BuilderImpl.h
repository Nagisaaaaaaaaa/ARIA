#pragma once

#include "ARIA/Tup.h"

namespace ARIA {

#define __ARIA_BUILDER_BEGIN(ACCESS, TYPE, BUILDER_NAME, N_PARTS)                                                      \
                                                                                                                       \
private:                                                                                                               \
  template <type_array::detail::ArrayType TUVWArray>                                                                   \
  class ARIA_CONCAT(BuilderImpl, BUILDER_NAME);                                                                        \
                                                                                                                       \
  ACCESS:                                                                                                              \
  using BUILDER_NAME = ARIA_CONCAT(BuilderImpl, BUILDER_NAME)<to_type_array_t<TecConstant<(N_PARTS), false>>>;         \
                                                                                                                       \
private:                                                                                                               \
  template <type_array::detail::ArrayType TUVWArray>                                                                   \
  class ARIA_CONCAT(BuilderImpl, BUILDER_NAME) {                                                                       \
  private:                                                                                                             \
    template <type_array::detail::ArrayType TUVWArray1>                                                                \
    using Impl = ARIA_CONCAT(BuilderImpl, BUILDER_NAME)<TUVWArray1>;                                                   \
                                                                                                                       \
  private:                                                                                                             \
    TYPE v_;                                                                                                           \
                                                                                                                       \
    [[nodiscard]] constexpr const TYPE &get() const {                                                                  \
      return v_;                                                                                                       \
    };                                                                                                                 \
                                                                                                                       \
    [[nodiscard]] constexpr TYPE &get() {                                                                              \
      return v_;                                                                                                       \
    };                                                                                                                 \
                                                                                                                       \
  private:                                                                                                             \
    template <type_array::detail::ArrayType TUVWArray1>                                                                \
    friend class ARIA_CONCAT(BuilderImpl, BUILDER_NAME);                                                               \
                                                                                                                       \
    explicit ARIA_CONCAT(BuilderImpl, BUILDER_NAME)(TYPE && v) : v_(std::move(v)) {}                                   \
                                                                                                                       \
  public:                                                                                                              \
    ARIA_CONCAT(BuilderImpl, BUILDER_NAME)() = default;                                                                \
                                                                                                                       \
  public:                                                                                                              \
    [[nodiscard]] constexpr operator TYPE() const & {                                                                  \
      static_assert(!TUVWArray::template has<C<false>>, "The instance has not been fully built");                      \
      return get();                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr operator TYPE() const && {                                                                 \
      static_assert(!TUVWArray::template has<C<false>>, "The instance has not been fully built");                      \
      return get();                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr operator TYPE() & {                                                                        \
      static_assert(!TUVWArray::template has<C<false>>, "The instance has not been fully built");                      \
      return get();                                                                                                    \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr operator TYPE() && {                                                                       \
      static_assert(!TUVWArray::template has<C<false>>, "The instance has not been fully built");                      \
      return std::move(get());                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr TYPE Build() const {                                                                       \
      return operator TYPE();                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr TYPE Build() {                                                                             \
      return operator TYPE();                                                                                          \
    }                                                                                                                  \
                                                                                                                       \
  private:                                                                                                             \
    class ARIA_CONCAT(DummyClassForBuilderBegin, BUILDER_NAME) {}

#define __ARIA_BUILDER_END                                                                                             \
  }                                                                                                                    \
  ;                                                                                                                    \
                                                                                                                       \
private:                                                                                                               \
  class ARIA_ANON(DummyClassForBuilderEnd) {}

//
//
//
#define __ARIA_BUILDER_MARK_PARAMS1(I)                                                                                 \
                                                                                                                       \
  using TUVWStatus = typename TUVWArray::template Get<(I)>;                                                            \
  static_assert(std::is_same_v<TUVWStatus, C<false>>, "Marked twice");                                                 \
                                                                                                                       \
  using TUVWArrayInserted = typename TUVWArray::template Insert<(I), C<true>>;                                         \
  using TUVWArrayInsertedErased = typename TUVWArrayInserted::template Erase<(I) + 1>;                                 \
                                                                                                                       \
  return Impl<TUVWArrayInsertedErased>{std::move(get())};

} // namespace ARIA
