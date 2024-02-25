#pragma once

#include "ARIA/Property.h"

namespace ARIA {

#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_STD_STRING(specifiers, type)                                           \
                                                                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., assign);                                                                       \
  /*ARIA_PROP_FUNC(public, specifiers, ., assign_range);*/                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., get_allocator);                                                                \
  /* Element access. */                                                                                                \
  ARIA_PROP_FUNC(public, specifiers, ., at);                                                                           \
  ARIA_PROP_FUNC(public, specifiers, ., front);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., back);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., data);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., c_str);                                                                        \
  /* Iterators. */                                                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., begin);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., cbegin);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., end);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., cend);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., rbegin);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., crbegin);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., rend);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., crend);                                                                        \
  /* Capacity. */                                                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., empty);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., size);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., length);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., max_size);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., reserve);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., capacity);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., shrink_to_fit);                                                                \
  /* Modifiers. */                                                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., clear);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., insert);                                                                       \
  /*ARIA_PROP_FUNC(public, specifiers, ., insert_range);*/                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., erase);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., push_back);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., pop_back);                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., append);                                                                       \
  /*ARIA_PROP_FUNC(public, specifiers, ., append_range);*/                                                             \
  ARIA_PROP_FUNC(public, specifiers, ., replace);                                                                      \
  /*ARIA_PROP_FUNC(public, specifiers, ., replace_with_range);*/                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., copy);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., resize);                                                                       \
  /*ARIA_PROP_FUNC(public, specifiers, ., resize_and_overwrite);*/                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., swap);                                                                         \
  /* Search. */                                                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., find);                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., rfind);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., find_first_of);                                                                \
  ARIA_PROP_FUNC(public, specifiers, ., find_first_not_of);                                                            \
  ARIA_PROP_FUNC(public, specifiers, ., find_last_of);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., find_last_not_of);                                                             \
  /* Operations. */                                                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., compare);                                                                      \
  ARIA_PROP_FUNC(public, specifiers, ., starts_with);                                                                  \
  ARIA_PROP_FUNC(public, specifiers, ., ends_with);                                                                    \
  /*ARIA_PROP_FUNC(public, specifiers, ., contains);*/                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., substr)

#define __ARIA_PROP_PREFAB_STD_STRING(accessGet, accessSet, specifiers, type, propName)                                \
  static_assert(std::is_same_v<std::decay_t<type>, std::string>,                                                       \
                "Type of the property should be `std::string` in order to use this prefab");                           \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_STD_STRING(specifiers, type);                                                \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_STD_STRING(specifiers, type, propName)                                                  \
  static_assert(std::is_same_v<std::decay_t<type>, std::string>,                                                       \
                "Type of the property should be `std::string` in order to use this prefab");                           \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_STD_STRING(specifiers, type);                                                \
  ARIA_PROP_END

} // namespace ARIA
