#include "GLFWInput.h"


GLFWInput::GLFWInput(GLFWwindow *window)
    : m_window(window){}

GLFWInput::~GLFWInput() {
    m_window = nullptr;
}

void GLFWInput::update() {
    if (!m_window)
        return;
    for (auto [act, key]: m_bind) {
        const bool pressed = glfwGetKey(m_window, key) == GLFW_PRESS;
        const bool hold = glfwGetKey(m_window, key) == GLFW_REPEAT;
        if (pressed || hold) m_curr = act;
    }
}
