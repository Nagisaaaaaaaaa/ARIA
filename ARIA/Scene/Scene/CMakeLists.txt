aria_setup_module(Scene Scene)

CPMAddPackage(
    NAME boost-assert
    GITHUB_REPOSITORY boostorg/assert
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-bind
    GITHUB_REPOSITORY boostorg/bind
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-concept-check
    GITHUB_REPOSITORY boostorg/concept_check
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-config
    GITHUB_REPOSITORY boostorg/config
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-container-hash
    GITHUB_REPOSITORY boostorg/container_hash
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-core
    GITHUB_REPOSITORY boostorg/core
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-describe
    GITHUB_REPOSITORY boostorg/describe
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-detail
    GITHUB_REPOSITORY boostorg/detail
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-function
    GITHUB_REPOSITORY boostorg/function
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-function-types
    GITHUB_REPOSITORY boostorg/function_types
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-functional
    GITHUB_REPOSITORY boostorg/functional
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-fusion
    GITHUB_REPOSITORY boostorg/fusion
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-io
    GITHUB_REPOSITORY boostorg/io
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-iterator
    GITHUB_REPOSITORY boostorg/iterator
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-move
    GITHUB_REPOSITORY boostorg/move
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-mp11
    GITHUB_REPOSITORY boostorg/mp11
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-mpl
    GITHUB_REPOSITORY boostorg/mpl
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-optional
    GITHUB_REPOSITORY boostorg/optional
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-predef
    GITHUB_REPOSITORY boostorg/predef
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-preprocessor
    GITHUB_REPOSITORY boostorg/preprocessor
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-smart-ptr
    GITHUB_REPOSITORY boostorg/smart_ptr
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-static-assert
    GITHUB_REPOSITORY boostorg/static_assert
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-throw-exception
    GITHUB_REPOSITORY boostorg/throw_exception
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-tuple
    GITHUB_REPOSITORY boostorg/tuple
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-type-traits
    GITHUB_REPOSITORY boostorg/type_traits
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-typeof
    GITHUB_REPOSITORY boostorg/typeof
    GIT_TAG boost-1.84.0
)

CPMAddPackage(
    NAME boost-utility
    GITHUB_REPOSITORY boostorg/utility
    GIT_TAG boost-1.84.0
)

set(scene_scene_headers
    ARIA/Scene/Components/Transform.h
    ARIA/Scene/Behavior.h
    ARIA/Scene/Component.h
    ARIA/Scene/Object.h
)

set(scene_scene_sources
    Components/Transform.cpp
    Component.cpp
    Object.cpp
)

aria_add_module(
    Scene Scene
    HEADERS ${scene_scene_headers}
    SOURCES ${scene_scene_sources}
    HARD_DEPENDENCIES
    ARIA::Core
    Boost::iterator
    CMAKE_SUBDIRS tests
)
