aria_setup_module(Scene Visual)

CPMAddPackage(
    NAME glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.4
)
add_subdirectory("${glad_SOURCE_DIR}/cmake" glad_cmake)
glad_add_library(glad_gl_core_mx_43 REPRODUCIBLE MX API gl:core=4.3)

CPMAddPackage(
    NAME glfw
    GIT_TAG 3.3.9
    GITHUB_REPOSITORY glfw/glfw
    OPTIONS
    "GLFW_BUILD_DOCS OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_EXAMPLES OFF"
)

find_package(Vulkan REQUIRED)

set(scene_visual_headers
    ARIA/Scene/Components/Camera.h
    ARIA/Scene/Components/LineRenderer.h
    ARIA/Scene/Components/Renderer.h
    ARIA/Scene/Components/TriMeshRenderer.h
    ARIA/Scene/Components/Window.h
    ARIA/Scene/Material.h
    ARIA/Scene/Shader.h
)

set(scene_visual_sources
    Camera.cpp
    Window.cpp
)

aria_add_module(
    Scene Visual
    HEADERS ${scene_visual_headers}
    SOURCES ${scene_visual_sources}
    HARD_DEPENDENCIES
    ARIA::Core
    ARIA::Scene::Scene
    glad_gl_core_mx_43
    glfw
    Vulkan::Vulkan
    CMAKE_SUBDIRS tests
)
