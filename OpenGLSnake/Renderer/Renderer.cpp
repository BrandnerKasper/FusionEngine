#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

#include "Renderer.h"
#include "../settings.h"

Renderer::Renderer(GLFWwindow* window)
    : m_window{window}{
    init();
}

void Renderer::init() {
    constexpr auto size = static_cast<float>(Settings::Game::board_size);
    camera = glm::ortho(0.0f, size, size, 0.0f, -1.0f, 1.0f);
    initSprites();
}

Renderer::~Renderer() {
    m_window = nullptr;
}

void Renderer::draw(std::string_view board) {
    updateSprites(board);

    m_render_texture.begin();
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT); // | GL_DEPTH_BUFFER_BIT

    for (const auto& sprite: m_sprites) {
        sprite.draw(camera);
    }


    int winW, winH;
    glfwGetFramebufferSize(m_window, &winW, &winH);
    m_render_texture.end(winW, winH);
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_render_texture.draw();

    m_render_texture.getTextureImage();

    // swap and check
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void Renderer::initSprites() {
    constexpr int size{Settings::Game::board_size};
    for (int i{0}; i < size; ++i) {
        for (int j{0}; j < size; ++j) {
            m_sprites.emplace_back(
                glm::vec2{j, i},
                "#000000");
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

