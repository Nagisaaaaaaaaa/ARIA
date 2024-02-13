#include "ARIA/Render/ScriptableRenderContext.h"
#include "ARIA/Scene/Components/Window.h"

#include <iostream>

namespace ARIA {

void ScriptableRenderContext::DrawRenderers() {
  // TODO: Draw all the renders according to the main camera and the main window.
  //  In the future, support multiple cameras and multiple windows

  std::cout << "Hello Render pipeline" << std::endl;

  Window::main()->Clear();
}

} // namespace ARIA
