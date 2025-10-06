#pragma once
#include <unordered_map>

#include <GLFW/glfw3.h>


class Input {
public:
    enum Action {Quit, Up, Down, Left, Right, Pause};

    static bool pressed(Action a);

private:
    static inline GLFWwindow* m_window = nullptr;
    static inline std::unordered_map<Action, int> m_bindings {
            {Action::Quit,  GLFW_KEY_ESCAPE},
            {Action::Up,    GLFW_KEY_W},
            {Action::Left,  GLFW_KEY_A},
            {Action::Down,  GLFW_KEY_S},
            {Action::Right, GLFW_KEY_D},
            {Action::Pause, GLFW_KEY_P},
        };

    friend class Application;
};