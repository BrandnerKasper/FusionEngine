#include <iostream>

#include "Application.h"

#include "Window/GLFWWindow.h"
#include "Input/GLFWInput.h"
#include "Input/TerminalInput.h"
#include "Renderer/ASCIIRenderer.h"
#include "Renderer/OpenGLRenderer.h"
#include "Renderer/NeuralRenderer.h"


Application::Application() {
    // initWindow();
    m_window = std::make_unique<GLFWWindow>();
    m_input = std::make_unique<GLFWInput>(m_window->handleAs<GLFWwindow>());
    m_renderer_map.emplace("ASCII", std::make_unique<ASCIIRenderer>());
    m_renderer_map.emplace("OpenGL", std::make_unique<OpenGLRenderer>(m_window->handleAs<GLFWwindow>()));
    m_renderer_map.emplace("Neural", std::make_unique<NeuralRenderer>(m_window->handleAs<GLFWwindow>()));
}

void Application::run() {
    while(m_window->shouldClose() && m_current_action != IInput::Quit) {
        // Delta Time
        const auto currentTime {glfwGetTime()};
        m_deltaTime = currentTime - m_last_frame;
        m_last_frame = currentTime;

        processInput();
        update();
        render();
        if (m_generate)
            genData();
    }
}

void Application::processInput() {
    m_prev_action = m_current_action;
    m_input->update();
    m_current_action = m_input->getAction();
}

void Application::update() {
    if (m_current_action == IInput::Pause)
        return;
    if (m_current_action == IInput::Switch) {
        switchRenderer();
        m_current_action = m_prev_action;
    }
    if (auto play = m_game.run(m_deltaTime, m_current_action))
        board_state = m_game.getBoardState();
    else {
        m_input->clear();
        m_game.reset();
    }
}

void Application::render() {
    // Terminal render
    m_last_render += m_deltaTime;
    if (m_last_render >= Settings::Render::frame_time) {
        m_renderer_map["ASCII"]->draw(board_state);
        m_last_render -= Settings::Render::frame_time;
    }
    // Either OpenGL or Neural Renderer
    m_renderer_map[m_curr_renderer]->draw(board_state);
}

void Application::genData() {
    static int count {};
    if (count == Settings::Data::amount)
        return;
    if (prev_board_state != board_state) {
        prev_board_state = board_state;
        for (auto& [name, renderer]: m_renderer_map) {
            renderer->generateData(name, count);
        }
        ++count;
    }
}

void Application::switchRenderer() {
    if (m_curr_renderer == "OpenGL")
        m_curr_renderer = "Neural";
    else
        m_curr_renderer = "OpenGL";
    m_window->setTitle(m_curr_renderer);
}

