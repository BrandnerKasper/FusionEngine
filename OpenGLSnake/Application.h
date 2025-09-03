#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "settings.h"

#include "Input.h"
#include "Renderer/Renderer.h"
#include "Game.h"

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

private:
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};

    // Delta time
    double m_deltaTime {}, m_last_frame {};

    Input m_input;
    Renderer m_renderer;
    Game m_game;
};