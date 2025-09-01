#include <print>
#include <glad/glad.h>

#include "Game.h"


Game::Game(int width, int height)
    : m_width{width}, m_height{height} {
    initWindow();
}

Game::~Game() {
    glfwTerminate();
}

void Game::run() {
    while (!glfwWindowShouldClose(m_window)) {
        processInput();

        // rendering commands
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // swap and check
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

void Game::processInput() {
    if (glfwGetKey(m_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(m_window, true);
}


// OpenGL call backs
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void Game::initWindow() {
    glfwInit();

    // Define OpenGL version (4.6)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create our first window
    m_window = glfwCreateWindow(m_width, m_height, "Fusion Snake", nullptr, nullptr);
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
