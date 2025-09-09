#include "Input.h"


bool Input::pressed(const Action a) {
    if (!m_window)
        return false;
    const bool pressed = glfwGetKey(m_window, m_bindings[a]) == GLFW_PRESS;
    const bool hold = glfwGetKey(m_window, m_bindings[a]) == GLFW_REPEAT;
    // bool released = glfwGetKey(m_window, m_bindings[a]) == GLFW_RELEASE;

    return pressed || hold;
}
