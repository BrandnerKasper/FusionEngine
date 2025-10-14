#pragma once

// TODO get all OpenGL code out of the application class
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <string_view>

#include "settings.h"
#include "Input/IInput.h"
#include "Renderer/ASCIIRenderer.h"
#include "Renderer/OpenGLRenderer.h"
#include "Renderer/NeuralRenderer.h"
#include "Game/Game.h"


class Application {
public:
    Application();
    virtual ~Application() = default;

    void run();

private:
    void init();
    void processInput();
    void update();
    void render();
    void terminalRender(std::string_view board) const;
    void openGLRender(std::string_view board) const;
    void neuralRender(std::string_view board) const;
    void genData();

private:
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};

    // Delta time
    double m_deltaTime {}, m_last_frame {};

    std::unique_ptr<IInput> m_input;
    IInput::Action m_current_action {IInput::Pause};
    std::unique_ptr<IRenderer> m_opengl_renderer;
    std::unique_ptr<ASCIIRenderer> m_ascii_renderer;
    std::unique_ptr<NeuralRenderer> m_neural_renderer;

    Game m_game;
    std::string board_state {};
    std::string prev_board_state {};
    bool generate {Settings::Data::generate};

    double m_last_render {};
};