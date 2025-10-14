#pragma once

// TODO get all OpenGL code out of the application class
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <memory>

#include "settings.h"
#include "Input/IInput.h"
#include "Renderer/IRenderer.h"
#include "Game/Game.h"


class Application {
public:
    Application();
    virtual ~Application() = default;

    void run();

private:
    void initWindow();
    void processInput();
    void update();
    void render();
    void genData();

    void switchRenderer();
    void setWindowTitle(const std::string& sub) const;

private:
    // Window
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};

    // Delta time
    double m_deltaTime {}, m_last_frame {};

    // Input
    std::unique_ptr<IInput> m_input;
    IInput::Action m_current_action {IInput::Pause};
    IInput::Action m_prev_action {};

    // Rendering
    std::unordered_map<std::string, std::unique_ptr<IRenderer>> m_renderer_map;
    std::string m_curr_renderer {"OpenGL"};
    double m_last_render {};

    // Game
    Game m_game;
    std::string board_state {};
    std::string prev_board_state {};

    // Data
    bool m_generate {Settings::Data::generate};
};