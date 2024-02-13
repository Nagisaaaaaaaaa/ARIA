aria_setup_module(Core SanitizerGTestMain)

aria_add_module(
    Core SanitizerGTestMain
    HEADERS tests/Common.h
    SOURCES SanitizerGTestMain.cpp
    HARD_DEPENDENCIES
    gtest
    STATIC # No shared library for this.
    ENABLE_IF ARIA_BUILD_TESTS
)
