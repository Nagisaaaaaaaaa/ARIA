#include "ARIA/Scene/Components/Window.h"

namespace ARIA {

std::string Window::ARIA_PROP_IMPL(title)() const {
  return title_;
}

void Window::ARIA_PROP_IMPL(title)(const std::string &value) {
  title_ = value;

  //! Should not be `title().c_str()` here because `title()` will make a copy, and
  //! GLFW requires a pointer to an existing C-string.
  glfwSetWindowTitle(window_, title_.c_str());
}

Vec2u Window::ARIA_PROP_IMPL(size)() const {
  int x, y;
  glfwGetWindowSize(window_, &x, &y);
  return {x, y};
}

void Window::ARIA_PROP_IMPL(size)(const Vec2u &value) {
  glfwSetWindowSize(window_, value.x(), value.y());
  context_->Viewport(0, 0, value.x(), value.y());
}

//
//
//
bool Window::ARIA_PROP_IMPL(shouldClose)() const {
  return static_cast<bool>(glfwWindowShouldClose(window_));
}

void Window::ARIA_PROP_IMPL(shouldClose)(const bool &value) {
  glfwSetWindowShouldClose(window_, static_cast<int>(value));
}

//
//
//
Vec4r Window::ARIA_PROP_IMPL(clearColor)() const {
  Vec4f color;
  context_->GetFloatv(GL_COLOR_CLEAR_VALUE, color.data());
  return color;
}

void Window::ARIA_PROP_IMPL(clearColor)(const Vec4r &value) {
  context_->ClearColor(value.x(), value.y(), value.z(), value.w());
}

//
//
//
//
//
void Window::SwapBuffers() {
  glfwSwapBuffers(window_);
}

//
//
//
//
//
bool Window::GetKey(KeyCode keyCode) {
  return static_cast<bool>(glfwGetKey(window_, static_cast<int>(keyCode)));
}

bool Window::GetMouseButton(int mouseButton) {
  return static_cast<bool>(glfwGetMouseButton(window_, mouseButton));
}

//
//
//
//
//
Window *&Window::main() noexcept {
  return main_;
}

//
//
//
//
//
void Window::WaitEvents() {
  glfwWaitEvents();

  for (Window &w : Window::range()) {
    if (w.nKeysPressed_ > 0 || w.anyKeyReleasedThisFrame_) {
      w.keyEventStation().go();
      w.anyKeyReleasedThisFrame_ = false;
    }

    if (w.nMouseButtonsPressed_ > 0 || w.anyMouseButtonReleasedThisFrame_) {
      w.mouseButtonEventStation().go();
      w.anyMouseButtonReleasedThisFrame_ = false;
    }
  }
}

//
//
//
//
//
void Window::Clear() {
  context_->Clear(GL_COLOR_BUFFER_BIT);
}

//
//
//
//
//
Window::Window(Object &object) : Behavior(object) {
  glfwSetErrorCallback(
      [](int error, const char *description) { fprintf(stderr, "GLFW error %d: %s\n", error, description); });

  if (!glfwInit()) // Initialize glfw library.
    throw std::exception{};

  // Setting glfw window hints and global configurations.
  const Vec2i glVersion{4, 3};
  {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, glVersion.x());
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, glVersion.y());
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Use core mode.
    // glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE); // Use debug context.
    glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Uncomment this statement to fix compilation on OS X.
#endif
  }

  // Create a windowed mode window and its OpenGL context.
  window_ = glfwCreateWindow(1, 1, "", nullptr, nullptr);
  if (!window_) {
    glfwTerminate();
    throw std::exception{};
  }

  // Make the window's context current.
  glfwMakeContextCurrent(window_);

  // Enable vsync.
  glfwSwapInterval(1);

  // Create GL context.
  context_ = std::unique_ptr<GladGLContext>(static_cast<GladGLContext *>(calloc(1, sizeof(GladGLContext))));
  if (!context_) {
    glfwTerminate();
    throw std::exception{};
  }
  int const version = gladLoadGLContext(context_.get(), glfwGetProcAddress);
  if (GLAD_VERSION_MAJOR(version) != glVersion.x() || GLAD_VERSION_MINOR(version) != glVersion.y()) {
    glfwTerminate();
    throw std::exception{};
  }

  // Setup callback functions.
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow *window, int width, int height) {
    for (Window &w : Window::range()) {
      if (w.window_ == window) {
        w.size() = {width, height};
      }
    }
  });
  glfwSetKeyCallback(window_, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
    for (Window &w : Window::range()) {
      if (w.window_ == window) {
        if (action == GLFW_PRESS)
          ++w.nKeysPressed_;
        else if (action == GLFW_RELEASE) {
          --w.nKeysPressed_;
          w.anyKeyReleasedThisFrame_ = true;
        }
      }
    }
  });

  glfwSetMouseButtonCallback(window_, [](GLFWwindow *window, int button, int action, int mods) {
    for (Window &w : Window::range()) {
      if (w.window_ == window) {
        if (action == GLFW_PRESS)
          ++w.nMouseButtonsPressed_;
        else if (action == GLFW_RELEASE) {
          --w.nMouseButtonsPressed_;
          w.anyMouseButtonReleasedThisFrame_ = true;
        }
      }
    }
  });

  // glfwSetScrollCallback(window_, Input::CallbackScroll);

  // Set as the main window.
  main() = this;
}

Window::~Window() {
  // Detach main window.
  if (main() == this)
    main() = nullptr;

  // Destroy the underlying glfw window.
  glfwDestroyWindow(window_);

  // Terminate glfw if this is the only window.
  if (Registry<Window>::size() == 1) {
    glfwTerminate();
  }
}

//
//
//
//
//
Window *Window::main_{};

} // namespace ARIA
