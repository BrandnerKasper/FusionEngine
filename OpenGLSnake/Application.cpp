#include "Application.h"

#include <print>

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
    if(m_window == nullptr) {
        std::println("Failed to create GLFW window!");
        return;
    }

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::println("Failed to initialize GLAD");
        return;
    }
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
}

void Application::update() {
    m_game.update(m_deltaTime);
}

void Application::render() {
    m_renderer.draw();
}
