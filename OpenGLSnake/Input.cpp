#include "Input.h"


bool Input::pressed(const Action a) {
    if (!m_window)
        return false;
    return glfwGetKey(m_window, m_bindings[a]) == GLFW_PRESS;
}
