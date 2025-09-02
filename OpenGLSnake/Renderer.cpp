#include <print>
#include <glad/glad.h>

#include "Renderer.h"


void Renderer::draw() {
    // Render stuff
    // rendering commands
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // swap and check
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}
