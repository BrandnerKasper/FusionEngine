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
    m_shader.use();
    const auto size = static_cast<float>(Settings::Game::board_size);
    const glm::mat4 projection = glm::ortho(0.0f, size,
        size, 0.0f, -1.0f, 1.0f);
    m_shader.setValue("projection", projection);

    createBuffer();

    initSprites();
}

Renderer::~Renderer() {
    m_window = nullptr;
}

void Renderer::draw(std::string_view board) {
    updateSprites(board);

    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, 32, 32);
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT); // | GL_DEPTH_BUFFER_BIT

    m_shader.use();
    const auto size = static_cast<float>(Settings::Game::board_size);
    const glm::mat4 projection = glm::ortho(0.0f, size,
        size, 0.0f, -1.0f, 1.0f);
    m_shader.setValue("projection", projection);

    for (const auto& sprite: m_sprites) {
        sprite.draw();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    int winW, winH;
    glfwGetFramebufferSize(m_window, &winW, &winH);
    glViewport(0, 0, winW, winH);
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
                m_shader,
                std::vector{
                    0.0f, 1.0f, 0.0f, 1.0f,
                    1.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 0.0f,

                    0.0f, 1.0f, 0.0f, 1.0f,
                    1.0f, 1.0f, 1.0f, 1.0f,
                    1.0f, 0.0f, 1.0f, 0.0f
                },
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

void Renderer::createBuffer() {
    // parameter for nearest neighbour scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // FBO (no depth needed)
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_render_texture.m_colorTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Failed to initialize texture buffer");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}
