#include <print>

#include "GLFWInput.h"


GLFWInput::GLFWInput(GLFWwindow* window)
    : m_window(window) {}

GLFWInput::~GLFWInput() {
    m_window = nullptr;
}

void GLFWInput::update() {
    checkKeyBoard();
    checkGamePad();
}

void GLFWInput::checkKeyBoard() {
    if (!m_window)
        return;
    for (auto [act, key]: m_bind_key_board) {
        const bool pressed = glfwGetKey(m_window, key) == GLFW_PRESS;
        const bool hold = glfwGetKey(m_window, key) == GLFW_REPEAT;
        if (pressed || hold) m_curr = act;
    }
}

void GLFWInput::checkGamePad() {
    GLFWgamepadstate state{};

    if (glfwJoystickIsGamepad(m_jid) && glfwGetGamepadState(m_jid, &state)) {
        for (auto [act, key]: m_bind_game_pad) {
            if (state.buttons[key])
                m_curr = act;
        }
    }
}
