#include "ARIA/Life/Offline/Lifecycle.h"
#include "ARIA/Life/Offline/Destroy.h"
#include "ARIA/Life/Offline/MonoBehavior.h"
#include "ARIA/Render/RenderPipelineManager.h"
#include "ARIA/Scene/Components/Window.h"
#include "ARIA/Scene/Object.h"

#include <cppcoro/sync_wait.hpp>
#include <cppcoro/when_all.hpp>

#include <ranges>

namespace ARIA {

namespace detail {

// Type acts as a tag to find the correct `operator|` overload.
template <typename C>
struct to_helper {};

// This actually does the work.
template <typename Container, std::ranges::range R>
  requires std::convertible_to<std::ranges::range_value_t<R>, typename Container::value_type>
Container operator|(R &&r, to_helper<Container>) {
  return Container{r.begin(), r.end()};
}

} // namespace detail

// Couldn't find a concept for container, however a container is a range, but not a view.
template <std::ranges::range Container>
  requires(!std::ranges::view<Container>)
auto to() {
  return detail::to_helper<Container>{};
}

//
//
//
//
//
void OLifecycle::Boot() {
  std::vector evolutions = OMonoBehavior::range() |
                           std::views::transform([](OMonoBehavior &mb) { return mb.Evolve(); }) |
                           to<std::vector<cppcoro::task<>>>();

  evolutions.emplace_back([]() -> cppcoro::task<> {
    while (Window::main() && !Window::main()->shouldClose()) {
      RenderPipelineManager::Render();

      Window::main()->SwapBuffers();

      Window::WaitEvents();
      DestructAllDestroyed();
    }

    DestroyImmediate(*Window::main());

    co_return;
  }());

  cppcoro::sync_wait(cppcoro::when_all(std::move(evolutions)));
}

} // namespace ARIA
