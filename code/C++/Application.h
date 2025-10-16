#pragma once
#include <string>
#include <memory>

#include "settings.h"
#include "Window/IWindow.h"
#include "Input/IInput.h"
#include "Renderer/IRenderer.h"
#include "Game/Game.h"


class Application {
public:
    Application();
    virtual ~Application() = default;

    void run();

private:
    void processInput();
    void update();
    void render();
    void genData();
    void switchRenderer();

private:
    // Window
    std::unique_ptr<IWindow> m_window;

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
