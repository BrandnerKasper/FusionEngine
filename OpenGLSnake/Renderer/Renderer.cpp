#include <print>
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

    createTex();
    createNDC();

    initSprites();
}

Renderer::~Renderer() {
    m_window = nullptr;
}

void Renderer::draw(std::string_view board) {
    updateSprites(board);

    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_offW, m_offH);
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

    m_present.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    m_present.setValue("uTex", 0);

    glBindVertexArray(m_quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);

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

void Renderer::createTex() {
    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_offW, m_offH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // parameter for nearest neighbour scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // FBO (no depth needed)
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_colorTex, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("Failed to initialize texture buffer");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

float NDC_quad[] = {
    -1.f,-1.f,  0.f,0.f,
     1.f,-1.f,  1.f,0.f,
     1.f, 1.f,  1.f,1.f,
    -1.f,-1.f,  0.f,0.f,
     1.f, 1.f,  1.f,1.f,
    -1.f, 1.f,  0.f,1.f
};

void Renderer::createNDC() {
    glGenVertexArrays(1, &m_quadVAO);
    glGenBuffers(1, &m_quadVBO);
    glBindVertexArray(m_quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(NDC_quad), NDC_quad, GL_STATIC_DRAW);

    glad_glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));

    glBindVertexArray(0);
}
