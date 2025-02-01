aria_setup_module(Scene Scene)

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
    ARIA::Core::Math
    CMAKE_SUBDIRS tests
)
