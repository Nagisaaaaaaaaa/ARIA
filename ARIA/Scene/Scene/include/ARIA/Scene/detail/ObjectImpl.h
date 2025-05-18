#pragma once

#include "ARIA/Scene/Components/detail/TransformImpl.h"

namespace ARIA {

// TODO: Something like `object.parent()->parent()` is forbidden because
// clang does not allow the child class to have the same name with the parent class.
#define __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_OBJECT(specifiers, type)                                    \
                                                                                                                       \
  /* Name. */                                                                                                          \
  ARIA_SUB_PROP(specifiers, std::string, name);                                                                        \
                                                                                                                       \
  /* Parent. */                                                                                                        \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Object *, parent);*/                                                               \
  /**/ /* Name. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, std::string, name);*/                                                               \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, parent);*/                                                                \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, root);*/                                                                  \
  /**/ /* Transform. */                                                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform &, transform);*/                                                          \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Root. */                                                                                                          \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Object *, root);*/                                                                 \
  /**/ /* Name. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, std::string, name);*/                                                               \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, parent);*/                                                                \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, root);*/                                                                  \
  /**/ /* Transform. */                                                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform &, transform);*/                                                          \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Transform. */                                                                                                     \
  __ARIA_SUB_PROP_INCOMPLETE_PREFAB_TRANSFORM(specifiers, Transform &, transform);

//
//
//
#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_OBJECT(specifiers, type)                                               \
                                                                                                                       \
  /* Name. */                                                                                                          \
  ARIA_SUB_PROP(specifiers, std::string, name);                                                                        \
                                                                                                                       \
  /* Parent. */                                                                                                        \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Object *, parent);*/                                                               \
  /**/ /* Name. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, std::string, name);*/                                                               \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, parent);*/                                                                \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, root);*/                                                                  \
  /**/ /* Transform. */                                                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform &, transform);*/                                                          \
  /**/ /* Is root and is child of. */                                                                                  \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsRoot);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsChildOf);*/                                                           \
  /**/ /* Add and get component. */                                                                                    \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., AddComponent);*/                                                        \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., GetComponent);*/                                                        \
  /**/ /* Iterators. */                                                                                                \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., begin);*/                                                               \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., end);*/                                                                 \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., cbegin);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., cend);*/                                                                \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Root. */                                                                                                          \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Object *, root);*/                                                                 \
  /**/ /* Name. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, std::string, name);*/                                                               \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, parent);*/                                                                \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Object *, root);*/                                                                  \
  /**/ /* Transform. */                                                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform &, transform);*/                                                          \
  /**/ /* Is root and is child of. */                                                                                  \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsRoot);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsChildOf);*/                                                           \
  /**/ /* Add and get component. */                                                                                    \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., AddComponent);*/                                                        \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., GetComponent);*/                                                        \
  /**/ /* Iterators. */                                                                                                \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., begin);*/                                                               \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., end);*/                                                                 \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., cbegin);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., cend);*/                                                                \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Transform. */                                                                                                     \
  __ARIA_SUB_PROP_INCOMPLETE_PREFAB_TRANSFORM(specifiers, Transform &, transform);                                     \
  /* Is root and is child of. */                                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., IsRoot);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., IsChildOf);                                                                    \
  /* Add and get component. */                                                                                         \
  ARIA_PROP_FUNC(public, specifiers, ., AddComponent);                                                                 \
  ARIA_PROP_FUNC(public, specifiers, ., GetComponent);                                                                 \
  /* Iterators. */                                                                                                     \
  ARIA_PROP_FUNC(public, specifiers, ., begin);                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., end);                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., cbegin);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., cend);

//
//
//
//
//
#define __ARIA_PROP_INCOMPLETE_PREFAB_OBJECT(accessGet, accessSet, specifiers, type, propName)                         \
  static_assert(std::is_same_v<std::decay_t<type>, Object> || std::is_same_v<std::decay_t<type>, Object *>,            \
                "Type of the property should be `Object` or `Object*` in order to use this prefab");                   \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_OBJECT(specifiers, type);                                         \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_INCOMPLETE_PREFAB_OBJECT(specifiers, type, propName)                                           \
  static_assert(std::is_same_v<std::decay_t<type>, Object> || std::is_same_v<std::decay_t<type>, Object *>,            \
                "Type of the property should be `Object` or `Object*` in order to use this prefab");                   \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_OBJECT(specifiers, type);                                         \
  ARIA_PROP_END

//
//
//
#define __ARIA_PROP_PREFAB_OBJECT(accessGet, accessSet, specifiers, type, propName)                                    \
  static_assert(std::is_same_v<std::decay_t<type>, Object> || std::is_same_v<std::decay_t<type>, Object *>,            \
                "Type of the property should be `Object` or `Object*` in order to use this prefab");                   \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_OBJECT(specifiers, type);                                                    \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_OBJECT(specifiers, type, propName)                                                      \
  static_assert(std::is_same_v<std::decay_t<type>, Object> || std::is_same_v<std::decay_t<type>, Object *>,            \
                "Type of the property should be `Object` or `Object*` in order to use this prefab");                   \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_OBJECT(specifiers, type);                                                    \
  ARIA_PROP_END

} // namespace ARIA
