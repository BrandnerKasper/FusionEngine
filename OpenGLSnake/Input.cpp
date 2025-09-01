#include "Input.h"


bool Input::pressed(Action a) {
    const auto window = m_renderer->getWindow();
    return glfwGetKey(window, m_bindings[a]) == GLFW_PRESS;
}
