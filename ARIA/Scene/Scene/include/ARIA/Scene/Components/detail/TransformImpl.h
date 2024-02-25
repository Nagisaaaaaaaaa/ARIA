#pragma once

#include "ARIA/Mat.h"
#include "ARIA/Property.h"
#include "ARIA/Quat.h"
#include "ARIA/Vec.h"

namespace ARIA {

// TODO: Something like `trans.parent()->parent()` is forbidden because
// clang does not allow the child class to have the same name with the parent class.
#define __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_TRANSFORM(specifiers, type)                                 \
                                                                                                                       \
  /* Parent. */                                                                                                        \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Transform *, parent);*/                                                            \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, parent);*/                                                             \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, root);*/                                                               \
  /**/ /* Local position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localPosition);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, localRotation);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localScale);*/                                                               \
  /**/ /* Local euler angles. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);*/                                                         \
  /**/ /* Local 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localUp);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localDown);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localForward);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localBack);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localLeft);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localRight);*/                                                               \
  /**/ /* Transform matrices. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);*/                                                         \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);*/                                                          \
  /**/ /* World position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, position);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, rotation);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);*/                                                               \
  /**/ /* World 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, up);*/                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, down);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, forward);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, back);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, left);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, right);*/                                                                    \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Root. */                                                                                                          \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Transform *, root);*/                                                              \
  /**/ /* Parent. */                                                                                                   \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, parent);*/                                                             \
  /**/ /* Root. */                                                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, root);*/                                                               \
  /**/ /* Local position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localPosition);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, localRotation);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localScale);*/                                                               \
  /**/ /* Local euler angles. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);*/                                                         \
  /**/ /* Local 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localUp);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localDown);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localForward);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localBack);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localLeft);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localRight);*/                                                               \
  /**/ /* Transform matrices. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);*/                                                         \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);*/                                                          \
  /**/ /* World position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, position);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, rotation);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);*/                                                               \
  /**/ /* World 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, up);*/                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, down);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, forward);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, back);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, left);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, right);*/                                                                    \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Local position, rotation, and scale. */                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, localPosition);                                                                     \
  ARIA_SUB_PROP(specifiers, Quatr, localRotation);                                                                     \
  ARIA_SUB_PROP(specifiers, Vec3r, localScale);                                                                        \
  /* Local euler angles. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);                                                                  \
  /* Local 6 directions. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, localUp);                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, localDown);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localForward);                                                                      \
  ARIA_SUB_PROP(specifiers, Vec3r, localBack);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localLeft);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localRight);                                                                        \
  /* Transform matrices. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);                                                                  \
  ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);                                                                   \
  /* World position, rotation, and scale. */                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, position);                                                                          \
  ARIA_SUB_PROP(specifiers, Quatr, rotation);                                                                          \
  ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);                                                                        \
  /* World 6 directions. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, up);                                                                                \
  ARIA_SUB_PROP(specifiers, Vec3r, down);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, forward);                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, back);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, left);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, right)

//
//
//
#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_TRANSFORM(specifiers, type)                                            \
                                                                                                                       \
  /* Parent. */                                                                                                        \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Transform *, parent);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, parent);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, root);*/                                                               \
  /**/ /* Local position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localPosition);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, localRotation);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localScale);*/                                                               \
  /**/ /* Local euler angles. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);*/                                                         \
  /**/ /* Local 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localUp);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localDown);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localForward);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localBack);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localLeft);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localRight);*/                                                               \
  /**/ /* Transform matrices. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);*/                                                         \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);*/                                                          \
  /**/ /* World position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, position);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, rotation);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);*/                                                               \
  /**/ /* World 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, up);*/                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, down);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, forward);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, back);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, left);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, right);*/                                                                    \
  /**/ /* Is root. */                                                                                                  \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsRoot);*/                                                              \
  /**/ /* Transform point, vector, and direction. */                                                                   \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformPoint);*/                                                      \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformVector);*/                                                     \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformDirection);*/                                                  \
  /**/ /* Translate, rotate, and rotate around. */                                                                     \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., Translate);*/                                                           \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., Rotate);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., RotateAround);*/                                                        \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Root. */                                                                                                          \
  /*ARIA_SUB_PROP_BEGIN(specifiers, Transform *, root);*/                                                              \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, parent);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Transform *, root);*/                                                               \
  /**/ /* Local position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localPosition);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, localRotation);*/                                                            \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localScale);*/                                                               \
  /**/ /* Local euler angles. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);*/                                                         \
  /**/ /* Local 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localUp);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localDown);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localForward);*/                                                             \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localBack);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localLeft);*/                                                                \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, localRight);*/                                                               \
  /**/ /* Transform matrices. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);*/                                                         \
  /**/ /*ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);*/                                                          \
  /**/ /* World position, rotation, and scale. */                                                                      \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, position);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Quatr, rotation);*/                                                                 \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);*/                                                               \
  /**/ /* World 6 directions. */                                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, up);*/                                                                       \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, down);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, forward);*/                                                                  \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, back);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, left);*/                                                                     \
  /**/ /*ARIA_SUB_PROP(specifiers, Vec3r, right);*/                                                                    \
  /**/ /* Is root. */                                                                                                  \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., IsRoot);*/                                                              \
  /**/ /* Transform point, vector, and direction. */                                                                   \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformPoint);*/                                                      \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformVector);*/                                                     \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., TransformDirection);*/                                                  \
  /**/ /* Translate, rotate, and rotate around. */                                                                     \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., Translate);*/                                                           \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., Rotate);*/                                                              \
  /**/ /*ARIA_PROP_FUNC(public, specifiers, ., RotateAround);*/                                                        \
  /*ARIA_SUB_PROP_END;*/                                                                                               \
                                                                                                                       \
  /* Local position, rotation, and scale. */                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, localPosition);                                                                     \
  ARIA_SUB_PROP(specifiers, Quatr, localRotation);                                                                     \
  ARIA_SUB_PROP(specifiers, Vec3r, localScale);                                                                        \
  /* Local euler angles. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, localEulerAngles);                                                                  \
  /* Local 6 directions. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, localUp);                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, localDown);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localForward);                                                                      \
  ARIA_SUB_PROP(specifiers, Vec3r, localBack);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localLeft);                                                                         \
  ARIA_SUB_PROP(specifiers, Vec3r, localRight);                                                                        \
  /* Transform matrices. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Mat4r, localToParentMat);                                                                  \
  ARIA_SUB_PROP(specifiers, Mat4r, localToWorldMat);                                                                   \
  /* World position, rotation, and scale. */                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, position);                                                                          \
  ARIA_SUB_PROP(specifiers, Quatr, rotation);                                                                          \
  ARIA_SUB_PROP(specifiers, Vec3r, lossyScale);                                                                        \
  /* World 6 directions. */                                                                                            \
  ARIA_SUB_PROP(specifiers, Vec3r, up);                                                                                \
  ARIA_SUB_PROP(specifiers, Vec3r, down);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, forward);                                                                           \
  ARIA_SUB_PROP(specifiers, Vec3r, back);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, left);                                                                              \
  ARIA_SUB_PROP(specifiers, Vec3r, right);                                                                             \
  /* Is root. */                                                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., IsRoot);                                                                       \
  /* Transform point, vector, and direction. */                                                                        \
  ARIA_PROP_FUNC(public, specifiers, ., TransformPoint);                                                               \
  ARIA_PROP_FUNC(public, specifiers, ., TransformVector);                                                              \
  ARIA_PROP_FUNC(public, specifiers, ., TransformDirection);                                                           \
  /* Translate, rotate, and rotate around. */                                                                          \
  ARIA_PROP_FUNC(public, specifiers, ., Translate);                                                                    \
  ARIA_PROP_FUNC(public, specifiers, ., Rotate);                                                                       \
  ARIA_PROP_FUNC(public, specifiers, ., RotateAround)

//
//
//
//
//
#define __ARIA_PROP_INCOMPLETE_PREFAB_TRANSFORM(accessGet, accessSet, specifiers, type, propName)                      \
  static_assert(std::is_same_v<std::decay_t<type>, Transform> || std::is_same_v<std::decay_t<type>, Transform *>,      \
                "Type of the property should be `Transform` or `Transform*` in order to use this prefab");             \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_TRANSFORM(specifiers, type);                                      \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_INCOMPLETE_PREFAB_TRANSFORM(specifiers, type, propName)                                        \
  static_assert(std::is_same_v<std::decay_t<type>, Transform> || std::is_same_v<std::decay_t<type>, Transform *>,      \
                "Type of the property should be `Transform` or `Transform*` in order to use this prefab");             \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_INCOMPLETE_PREFAB_MEMBERS_TRANSFORM(specifiers, type);                                      \
  ARIA_PROP_END

//
//
//
#define __ARIA_PROP_PREFAB_TRANSFORM(accessGet, accessSet, specifiers, type, propName)                                 \
  static_assert(std::is_same_v<std::decay_t<type>, Transform> || std::is_same_v<std::decay_t<type>, Transform *>,      \
                "Type of the property should be `Transform` or `Transform*` in order to use this prefab");             \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, propName);                                                   \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_TRANSFORM(specifiers, type);                                                 \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_TRANSFORM(specifiers, type, propName)                                                   \
  static_assert(std::is_same_v<std::decay_t<type>, Transform> || std::is_same_v<std::decay_t<type>, Transform *>,      \
                "Type of the property should be `Transform` or `Transform*` in order to use this prefab");             \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, propName);                                                                     \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_TRANSFORM(specifiers, type);                                                 \
  ARIA_PROP_END

} // namespace ARIA
