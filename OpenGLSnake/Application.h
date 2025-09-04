#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <unordered_map>

#include "settings.h"

#include "Input.h"
#include "Renderer/Renderer.h"
#include "Game/Game.h"

class Application {
public:
    Application();
    virtual ~Application();

    void run();

private:
    void init();
    void processInput();
    void update();
    void render();
    void terminal_render();

private:
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};

    // Delta time
    double m_deltaTime {}, m_last_frame {};

    Input m_input;
    Input::Action m_current_action{Input::Up};
    Renderer m_renderer;
    Game m_game;

    double m_last_render {};

    // terminal render
    std::unordered_map<char, std::string> ascii {
        {'0', " "},
        {'1', "#"},
        {'2', "■"},
        {'3', "▫"}
    };
};