#pragma once

#include "ARIA/Concurrency/Registry.h"
#include "ARIA/Coro/Station.h"
#include "ARIA/PropertySTL.h"
#include "ARIA/Scene/Behavior.h"
#include "ARIA/Vec.h"

#include <glad/gl.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <span>

namespace ARIA {

enum class KeyCode : int {
  Unknown = GLFW_KEY_UNKNOWN,
  Space = GLFW_KEY_SPACE,
  Apostrophe = GLFW_KEY_APOSTROPHE, /* ' */
  Comma = GLFW_KEY_COMMA,           /* , */
  Minus = GLFW_KEY_MINUS,           /* - */
  Period = GLFW_KEY_PERIOD,         /* . */
  Slash = GLFW_KEY_SLASH,           /* / */
  Num0 = GLFW_KEY_0,
  Num1 = GLFW_KEY_1,
  Num2 = GLFW_KEY_2,
  Num3 = GLFW_KEY_3,
  Num4 = GLFW_KEY_4,
  Num5 = GLFW_KEY_5,
  Num6 = GLFW_KEY_6,
  Num7 = GLFW_KEY_7,
  Num8 = GLFW_KEY_8,
  Num9 = GLFW_KEY_9,
  Semicolon = GLFW_KEY_SEMICOLON, /* ; */
  Equal = GLFW_KEY_EQUAL,         /* = */
  A = GLFW_KEY_A,
  B = GLFW_KEY_B,
  C = GLFW_KEY_C,
  D = GLFW_KEY_D,
  E = GLFW_KEY_E,
  F = GLFW_KEY_F,
  G = GLFW_KEY_G,
  H = GLFW_KEY_H,
  I = GLFW_KEY_I,
  J = GLFW_KEY_J,
  K = GLFW_KEY_K,
  L = GLFW_KEY_L,
  M = GLFW_KEY_M,
  N = GLFW_KEY_N,
  O = GLFW_KEY_O,
  P = GLFW_KEY_P,
  Q = GLFW_KEY_Q,
  R = GLFW_KEY_R,
  S = GLFW_KEY_S,
  T = GLFW_KEY_T,
  U = GLFW_KEY_U,
  V = GLFW_KEY_V,
  W = GLFW_KEY_W,
  X = GLFW_KEY_X,
  Y = GLFW_KEY_Y,
  Z = GLFW_KEY_Z,
  LeftBracket = GLFW_KEY_LEFT_BRACKET,   /* [ */
  Backslash = GLFW_KEY_BACKSLASH,        /* \ */
  RightBracket = GLFW_KEY_RIGHT_BRACKET, /* ] */
  GraveAccent = GLFW_KEY_GRAVE_ACCENT,   /* ` */
  World1 = GLFW_KEY_WORLD_1,             /* non-US #1 */
  World2 = GLFW_KEY_WORLD_2,             /* non-US #2 */

  /* Function keys */
  Escape = GLFW_KEY_ESCAPE,
  Enter = GLFW_KEY_ENTER,
  Tab = GLFW_KEY_TAB,
  Backspace = GLFW_KEY_BACKSPACE,
  Insert = GLFW_KEY_INSERT,
  Delete = GLFW_KEY_DELETE,
  Right = GLFW_KEY_RIGHT,
  Left = GLFW_KEY_LEFT,
  Down = GLFW_KEY_DOWN,
  Up = GLFW_KEY_UP,
  PageUp = GLFW_KEY_PAGE_UP,
  PageDown = GLFW_KEY_PAGE_DOWN,
  Home = GLFW_KEY_HOME,
  End = GLFW_KEY_END,
  CapsLock = GLFW_KEY_CAPS_LOCK,
  ScrollLock = GLFW_KEY_SCROLL_LOCK,
  NumLock = GLFW_KEY_NUM_LOCK,
  PrintScreen = GLFW_KEY_PRINT_SCREEN,
  Pause = GLFW_KEY_PAUSE,
  F1 = GLFW_KEY_F1,
  F2 = GLFW_KEY_F2,
  F3 = GLFW_KEY_F3,
  F4 = GLFW_KEY_F4,
  F5 = GLFW_KEY_F5,
  F6 = GLFW_KEY_F6,
  F7 = GLFW_KEY_F7,
  F8 = GLFW_KEY_F8,
  F9 = GLFW_KEY_F9,
  F10 = GLFW_KEY_F10,
  F11 = GLFW_KEY_F11,
  F12 = GLFW_KEY_F12,
  F13 = GLFW_KEY_F13,
  F14 = GLFW_KEY_F14,
  F15 = GLFW_KEY_F15,
  F16 = GLFW_KEY_F16,
  F17 = GLFW_KEY_F17,
  F18 = GLFW_KEY_F18,
  F19 = GLFW_KEY_F19,
  F20 = GLFW_KEY_F20,
  F21 = GLFW_KEY_F21,
  F22 = GLFW_KEY_F22,
  F23 = GLFW_KEY_F23,
  F24 = GLFW_KEY_F24,
  F25 = GLFW_KEY_F25,
  KP_0 = GLFW_KEY_KP_0,
  KP_1 = GLFW_KEY_KP_1,
  KP_2 = GLFW_KEY_KP_2,
  KP_3 = GLFW_KEY_KP_3,
  KP_4 = GLFW_KEY_KP_4,
  KP_5 = GLFW_KEY_KP_5,
  KP_6 = GLFW_KEY_KP_6,
  KP_7 = GLFW_KEY_KP_7,
  KP_8 = GLFW_KEY_KP_8,
  KP_9 = GLFW_KEY_KP_9,
  KP_Decimal = GLFW_KEY_KP_DECIMAL,
  KP_Divide = GLFW_KEY_KP_DIVIDE,
  KP_Multiply = GLFW_KEY_KP_MULTIPLY,
  KP_Subtract = GLFW_KEY_KP_SUBTRACT,
  KP_Add = GLFW_KEY_KP_ADD,
  KP_Enter = GLFW_KEY_KP_ENTER,
  KP_Equal = GLFW_KEY_KP_EQUAL,
  LeftShift = GLFW_KEY_LEFT_SHIFT,
  LeftControl = GLFW_KEY_LEFT_CONTROL,
  LeftAlt = GLFW_KEY_LEFT_ALT,
  LeftSuper = GLFW_KEY_LEFT_SUPER,
  RightShift = GLFW_KEY_RIGHT_SHIFT,
  RightControl = GLFW_KEY_RIGHT_CONTROL,
  RightAlt = GLFW_KEY_RIGHT_ALT,
  RightSuper = GLFW_KEY_RIGHT_SUPER,
  Menu = GLFW_KEY_MENU,
};

//
//
//
//
//
/// \brief A `Window` is a device through which the player play the game.
class Window final : public Behavior, public Registry<Window> {
public:
  /// \brief Title of the window.
  ARIA_PROP_PREFAB_STD_STRING(public, public, , std::string, title)

  /// \brief Size of the window.
  /// `size().x()` is the width, and `size().y()` is the height.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec2u, size)

  //
  //
  //
  /// \brief When the user attempts to close the window, for example by clicking the close widget or
  /// using a key chord like Alt+F4, the close flag of the window is set.
  /// The window is however not actually destroyed and,
  /// unless you watch for this state change, nothing further happens.
  ARIA_PROP(public, public, , bool, shouldClose)

  //
  //
  //
  /// \brief Get the coroutine station for key event.
  ARIA_FWD_REF_PROP(public, , keyEventStation, keyEventStation_)

  /// \brief Get the coroutine station for mouse button event.
  ARIA_FWD_REF_PROP(public, , mouseButtonEventStation, mouseButtonEventStation_)

  //
  //
  //
  ARIA_PROP(public, public, , Vec4r, clearColor)

  //
  //
  //
public:
  /// \brief This function swaps the front and back buffers of the specified window when
  /// rendering with OpenGL or OpenGL ES.
  /// If the swap interval is greater than zero, the GPU driver
  /// waits the specified number of screen updates before swapping the buffers.
  void SwapBuffers();

  //
  //
  //
  /// \brief Returns true while the user holds down the key.
  bool GetKey(KeyCode keyCode);

  /// \brief Returns whether the given mouse button is held down.
  bool GetMouseButton(int mouseButton);

  //
  //
  //
public:
  /// \brief Get the main window.
  static Window *&main() noexcept;

  //
  //
  //
  /// \brief Puts the thread to sleep until at least one event has been received and
  /// then processes all received events.
  static void WaitEvents();

  //
  //
  //
public:
  void Clear();

  //
  //
  //
  //
  //
private:
  friend Object;

  explicit Window(Object &object);

public:
  Window(const Window &) = delete;
  Window(Window &&) noexcept = delete;
  Window &operator=(const Window &) = delete;
  Window &operator=(Window &&) noexcept = delete;
  ~Window() final;

  //
  //
  //
  //
  //
private:
  static Window *main_;

  //
  //
  //
  std::string title_{};

  GLFWwindow *window_{};
  std::unique_ptr<GladGLContext> context_{};

  // Number of keys currently pressed, updated by the key callback function.
  unsigned nKeysPressed_{0};
  // Whether there's any key released this frame, updated by the key callback function.
  bool anyKeyReleasedThisFrame_{false};
  // The coroutine station for key event.
  Coro::station keyEventStation_{};

  unsigned nMouseButtonsPressed_{0};
  bool anyMouseButtonReleasedThisFrame_{false};
  Coro::station mouseButtonEventStation_{};

  //
  //
  //
  //
  //
  //
  //
  //
  //
private:
  [[nodiscard]] std::string ARIA_PROP_GETTER(title)() const;
  void ARIA_PROP_SETTER(title)(const std::string &value);
  [[nodiscard]] Vec2u ARIA_PROP_GETTER(size)() const;
  void ARIA_PROP_SETTER(size)(const Vec2u &value);

  //
  [[nodiscard]] bool ARIA_PROP_GETTER(shouldClose)() const;
  void ARIA_PROP_SETTER(shouldClose)(const bool &value);

  //
  [[nodiscard]] Vec4r ARIA_PROP_GETTER(clearColor)() const;
  void ARIA_PROP_SETTER(clearColor)(const Vec4r &value);
};

} // namespace ARIA
