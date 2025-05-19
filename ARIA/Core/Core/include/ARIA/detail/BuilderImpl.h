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
    using BuilderImpl = ARIA_CONCAT(BuilderImpl, BUILDER_NAME)<TUVWArray1>;                                            \
                                                                                                                       \
  public:                                                                                                              \
    ARIA_CONCAT(BuilderImpl, BUILDER_NAME)() = default;                                                                \
                                                                                                                       \
  private:                                                                                                             \
    TYPE type_;                                                                                                        \
                                                                                                                       \
    [[nodiscard]] constexpr const TYPE &get() const {                                                                  \
      return type_;                                                                                                    \
    };                                                                                                                 \
                                                                                                                       \
    [[nodiscard]] constexpr TYPE &get() {                                                                              \
      return type_;                                                                                                    \
    };                                                                                                                 \
                                                                                                                       \
    template <type_array::detail::ArrayType TUVWArray1>                                                                \
    friend class ARIA_CONCAT(BuilderImpl, BUILDER_NAME);                                                               \
                                                                                                                       \
    explicit ARIA_CONCAT(BuilderImpl, BUILDER_NAME)(TYPE && type) : type_(std::move(type)) {}                          \
                                                                                                                       \
  public:                                                                                                              \
    [[nodiscard]] constexpr operator TYPE() {                                                                          \
      static_assert(!TUVWArray::template has<C<false>>, "Partially built");                                            \
      return std::move(get());                                                                                         \
    }                                                                                                                  \
                                                                                                                       \
    [[nodiscard]] constexpr TYPE Build() {                                                                             \
      return static_cast<TYPE>(*this);                                                                                 \
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
  return BuilderImpl<TUVWArrayInsertedErased>{std::move(get())};

} // namespace ARIA
