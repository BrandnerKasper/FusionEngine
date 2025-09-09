#include <print>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

#include "Renderer.h"
#include "../settings.h"

Renderer::Renderer(GLFWwindow* window)
    : m_window{window}{
    m_shader.use();
    const auto size = static_cast<float>(Settings::Game::board_size);
    const glm::mat4 projection = glm::ortho(0.0f, size,
        size, 0.0f, -1.0f, 1.0f);
    m_shader.setValue("projection", projection);
    initSprites();
}

Renderer::~Renderer() {
    m_window = nullptr;
}

void Renderer::draw(std::string_view board) {
    // rendering commands
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    updateSprites(board);
    for (const auto& sprite: m_sprites) {
        sprite.draw();
    }

    // swap and check
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void Renderer::initSprites() {
    constexpr int size {Settings::Game::board_size};
    for (int i{0}; i < size; ++i) {
        for (int j{0}; j < size; ++j) {
            m_sprites.emplace_back(m_shader, glm::vec2{j, i}, "#000000");
        }
    }
}

void Renderer::updateSprites(std::string_view board) {
    size_t idx {0};
    for (const auto c: board) {
        if (c == '\n')
            continue;
        m_sprites[idx++].setColor(color_map[c]);
    }
}
