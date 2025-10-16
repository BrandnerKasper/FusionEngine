#include <memory>

#include "GLFWWindow.h"


GLFWWindow::GLFWWindow() {
    init();
}

GLFWWindow::~GLFWWindow() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void GLFWWindow::setTitle(const std::string &t) const {
    const auto title = m_title + " - " + t;
    glfwSetWindowTitle(m_window, title.c_str());
}

bool GLFWWindow::shouldClose() const {
    return !glfwWindowShouldClose(m_window);
}

// OpenGL call backs
void framebuffer_size_callback(GLFWwindow* window, const int width, const int height) {
    glViewport(0, 0, width, height);
}

void GLFWWindow::init() {
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
