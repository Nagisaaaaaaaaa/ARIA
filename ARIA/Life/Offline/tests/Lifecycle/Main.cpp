#include "ARIA/Life/Offline/Lifecycle.h"
#include "ARIA/Life/Offline/MonoBehavior.h"
#include "ARIA/Render/DefaultRenderPipeline.h"
#include "ARIA/Render/RenderPipelineManager.h"
#include "ARIA/Scene/Components/Window.h"
#include "ARIA/Scene/Object.h"

#include <cppcoro/when_all.hpp>

#include <iostream>

namespace ARIA {

namespace {

class OUndine final : public OMonoBehavior {
public:
  ARIA_FWD_REF_PROP(public, , window, window_)

public:
  cppcoro::task<> Evolve() final {
    window()->size() = {500, 300};
    window()->clearColor() = {0.75_R, 0.75_R, 0.75_R, 0.75_R};

    auto taskESC = [&]() mutable -> cppcoro::task<> {
      while (co_await window()->keyEventStation().schedule()) {
        if (window()->GetKey(KeyCode::Escape)) {
          window()->shouldClose() = true;
        }
      }
    };

    auto taskK = [&](int id) mutable -> cppcoro::task<> {
      std::array keys = {KeyCode::J, KeyCode::K, KeyCode::L};
      std::array keysStr = {"J", "K", "L"};

      while (co_await window()->keyEventStation().schedule()) {
        if (window()->GetKey(keys[id])) {
          std::cout << keysStr[id] << std::endl;
          window()->title() = keysStr[id];

          if (id == 0)
            window()->size() += Vec2u{1, 0};
          else if (id == 1)
            window()->size() += Vec2u{0, 1};
          else if (id == 2)
            window()->size() += Vec2u{1, 1};
        }
      }
    };

    auto taskM = [&](int id) mutable -> cppcoro::task<> {
      while (co_await window()->mouseButtonEventStation().schedule()) {
        if (window()->GetMouseButton(id)) {
          std::cout << "Mouse " << id << std::endl;
        }
      }
    };

    co_await cppcoro::when_all(taskESC(), taskK(0), taskK(1), taskK(2), taskM(0), taskM(1), taskM(2));
  }

public:
  explicit OUndine(Window *window) : window_(window) {}

  OUndine(const OUndine &) = delete;
  OUndine(OUndine &&) noexcept = delete;
  OUndine &operator=(const OUndine &) = delete;
  OUndine &operator=(OUndine &&) noexcept = delete;
  ~OUndine() final = default;

private:
  Window *window_;
};

//
//
//
//
//
int Main() {
  // Setup scene.
  Object &o0 = Object::Create();
  o0.name() = "o0";
  Object &o1 = Object::Create();
  o1.name() = "o1";
  o1.parent() = &o0;

  o0.AddComponent<Window>();

  // Setup offline mono behaviors.
  OUndine undine{o0.GetComponent<Window>()};

  // Setup Render pipeline.
  RenderPipelineManager::currentPipeline() = std::make_unique<DefaultRenderPipeline>();

  // Boot lifecycle.
  OLifecycle lifecycle;
  lifecycle.Boot();

  return 0;
}

} // namespace

} // namespace ARIA

//
//
//
//
//
int main() {
  try {
    return ARIA::Main();
  } catch (const std::exception &e) { std::cout << e.what() << std::endl; }

  return 0;
}
