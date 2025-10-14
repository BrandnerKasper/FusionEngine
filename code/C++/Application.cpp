#include <iostream>

#include "Application.h"

#include "Input/GLFWInput.h"
#include "Input/TerminalInput.h"
#include "Renderer/ASCIIRenderer.h"
#include "Renderer/OpenGLRenderer.h"
#include "Renderer/NeuralRenderer.h"


Application::Application() {
    initWindow();
    m_input = std::make_unique<GLFWInput>(m_window);
    m_renderer_map.emplace("ASCII", std::make_unique<ASCIIRenderer>());
    m_renderer_map.emplace("OpenGL", std::make_unique<OpenGLRenderer>(m_window));
    m_renderer_map.emplace("Neural", std::make_unique<NeuralRenderer>(m_window));
}

// OpenGL call backs
void framebuffer_size_callback(GLFWwindow* window, const int width, const int height) {
    glViewport(0, 0, width, height);
}

void Application::initWindow() {
    glfwInit();

    // Define OpenGL version (4.6)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create our first window
    m_window = glfwCreateWindow(m_width, m_height, (m_title+" - OpenGL").c_str(), nullptr, nullptr);
    if(m_window == nullptr)
        throw std::runtime_error("Failed to create GLFW window!");

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialize GLAD");
    // Depth testing
    glEnable(GL_DEPTH_TEST);
}

void Application::run() {
    while(!glfwWindowShouldClose(m_window) && m_current_action != IInput::Quit) {
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
    setWindowTitle(m_curr_renderer);
}

void Application::setWindowTitle(const std::string& sub) const {
    const auto t = m_title + " - " + sub;
    glfwSetWindowTitle(m_window, t.c_str());
}
