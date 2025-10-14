#pragma once
#include <GLFW/glfw3.h>
#include <unordered_map>

#include "IInput.h"



class GLFWInput final : public IInput{
public:
    explicit GLFWInput(GLFWwindow* window);
    ~GLFWInput() override;

    void update() override;

private:
    void checkKeyBoard();
    void checkGamePad();

private:
    GLFWwindow *m_window = nullptr;
    std::unordered_map<Action, int> m_bind_key_board{
        {Action::Quit, GLFW_KEY_ESCAPE},
        {Action::Up, GLFW_KEY_W},
        {Action::Left, GLFW_KEY_A},
        {Action::Down, GLFW_KEY_S},
        {Action::Right, GLFW_KEY_D},
        {Action::Pause, GLFW_KEY_P},
    };

    int m_jid = GLFW_JOYSTICK_1;
    std::unordered_map<Action, int> m_bind_game_pad {
            {Action::Quit,  GLFW_GAMEPAD_BUTTON_BACK},
            {Action::Up,    GLFW_GAMEPAD_BUTTON_DPAD_UP},
            {Action::Left,  GLFW_GAMEPAD_BUTTON_DPAD_LEFT},
            {Action::Down,  GLFW_GAMEPAD_BUTTON_DPAD_DOWN},
            {Action::Right, GLFW_GAMEPAD_BUTTON_DPAD_RIGHT},
            {Action::Pause, GLFW_GAMEPAD_BUTTON_START},
        };
};
