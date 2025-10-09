#pragma once
#include <unordered_map>

#include <GLFW/glfw3.h>

#include "IInput.h"


class GLFWInput final : public IInput{
public:
    explicit GLFWInput(GLFWwindow* window);
    ~GLFWInput() override;

    void update() override;

private:
    GLFWwindow* m_window = nullptr;
    std::unordered_map<Action, int> m_bind {
            {Action::Quit,  GLFW_KEY_ESCAPE},
            {Action::Up,    GLFW_KEY_W},
            {Action::Left,  GLFW_KEY_A},
            {Action::Down,  GLFW_KEY_S},
            {Action::Right, GLFW_KEY_D},
            {Action::Pause, GLFW_KEY_P},
        };
};
