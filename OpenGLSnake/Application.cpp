#include <iostream>

#include "Application.h"


Application::Application() {
    init();
    Input::m_window = m_window;
    Renderer::m_window = m_window;
}

Application::~Application() {
    Input::m_window = nullptr;
    Renderer::m_window = nullptr;
}

// OpenGL call backs
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Application::init() {
    glfwInit();

    // Define OpenGL version (4.6)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create our first window
    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if(m_window == nullptr)
        throw std::runtime_error("Failed to create GLFW window!");

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialize GLAD");
}

void Application::run() {
    while(!glfwWindowShouldClose(m_window)) {
        // Delta Time
        const auto currentTime {glfwGetTime()};
        m_deltaTime = currentTime - m_last_frame;
        m_last_frame = currentTime;

        processInput();
        update();
        render();
    }
}

void Application::processInput() {
    if (Input::pressed(Input::Quit))
        glfwSetWindowShouldClose(m_window, true);
    if (Input::pressed(Input::Pause))
        m_current_action = Input::Pause;
    // TODO EVENT SYSTEM to subscribe onto Input event
    if (Input::pressed(Input::Up))
        m_current_action = Input::Up;

    if (Input::pressed(Input::Down))
        m_current_action = Input::Down;

    if (Input::pressed(Input::Left))
        m_current_action = Input::Left;

    if (Input::pressed(Input::Right))
        m_current_action = Input::Right;
}

void Application::update() {
    if (m_current_action == Input::Pause)
        return;
    m_game.run(m_deltaTime, m_current_action);
}

void Application::render() {
    // Terminal render
    m_last_render += m_deltaTime;
    if (m_last_render >= Settings::Render::frame_time) {
        terminal_render();
        m_last_render -= Settings::Render::frame_time;
    }
    // OpenGL render
    m_renderer.draw();
}


void Application::terminal_render() {
    std::string terminal {};
    for (const auto c: m_game.getBoardState()) {
        if (c == '\n')
            terminal += c;
        else {
            terminal += ascii[c];
        }
    }
    std::cout << terminal << std::endl;
}
