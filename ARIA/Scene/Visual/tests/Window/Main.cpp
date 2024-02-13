#include "ARIA/Scene/Components/Window.h"
#include "ARIA/Scene/Object.h"

#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/when_all.hpp>
#include <gtest/gtest.h>

#include <iostream>

namespace ARIA {

namespace {

int Main() {
  Object &o = Object::Create();
  Window &window = o.AddComponent<Window>();

  cppcoro::static_thread_pool threadPool;

  auto windowTask = [&]() mutable -> cppcoro::task<> {
    while (!window.shouldClose()) {
      window.SwapBuffers();

      Window::WaitEvents();
    }

    co_return;
  };

  auto taskESC = [&]() mutable -> cppcoro::task<> {
    while (co_await window.keyEventStation().schedule()) {
      if (window.GetKey(KeyCode::Escape)) {
        window.shouldClose() = true;
      }
    }
  };

  auto taskK = [&](int id) mutable -> cppcoro::task<> {
    std::array keys = {KeyCode::J, KeyCode::K, KeyCode::L};
    std::array keysStr = {"J", "K", "L"};

    while (co_await window.keyEventStation().schedule()) {
      if (window.GetKey(keys[id])) {
        std::cout << keysStr[id] << std::endl;
        window.title() = keysStr[id];

        if (id == 0)
          window.size() += Vec2u{1, 0};
        else if (id == 1)
          window.size() += Vec2u{0, 1};
        else if (id == 2)
          window.size() += Vec2u{1, 1};
      }
    }
  };

  auto taskM = [&](int id) mutable -> cppcoro::task<> {
    while (co_await window.mouseButtonEventStation().schedule()) {
      if (window.GetMouseButton(id)) {
        std::cout << "Mouse " << id << std::endl;
      }
    }
  };

  cppcoro::sync_wait(
      cppcoro::when_all(taskESC(), taskK(0), taskK(1), taskK(2), taskM(0), taskM(1), taskM(2), windowTask()));

  return 0;
}

} // namespace

} // namespace ARIA

int main() {
  try {
    return ARIA::Main();
  } catch (const std::exception &e) { std::cout << e.what() << std::endl; }

  return 0;
}
