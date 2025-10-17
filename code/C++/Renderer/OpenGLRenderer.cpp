#include <glad/glad.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../extern/stb_image_write.h"

#include "OpenGLRenderer.h"
#include "../assets.h"


OpenGLRenderer::OpenGLRenderer(GLFWwindow* window)
    : m_window{window}{
    init();
}

void OpenGLRenderer::init() {
    initCamera();
    initSprites();
}

void OpenGLRenderer::initCamera() {
    constexpr auto size = static_cast<float>(Settings::Game::board_size);
    camera = glm::ortho(0.0f, size, size, 0.0f, -1.0f, 1.0f);
}

OpenGLRenderer::~OpenGLRenderer() {
    m_window = nullptr;
}

void OpenGLRenderer::draw(const std::string_view board) {
    updateSprites(board);

    glViewport(0, 0, m_size, m_size);
    m_render_texture.bind();
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT); // | GL_DEPTH_BUFFER_BIT

    for (const auto& sprite: m_sprites) {
        sprite.draw(camera);
    }


    int winW, winH;
    glfwGetFramebufferSize(m_window, &winW, &winH);
    glViewport(0, 0, winW, winH);
    m_render_texture.unbind();
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_render_texture.draw();

    // swap and check
    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

void saveTexAsImg(const std::string& filename, const int w, const int h, const std::vector<unsigned char>& rgba) {
    // Flip vertically for correct orientation
    std::vector<unsigned char> flipped(w * h * 4);
    for (int y = 0; y < h; ++y) {
        memcpy(&flipped[y * w * 4],
               &rgba[(h - 1 - y) * w * 4],
               w * 4);
    }

    stbi_write_png(filename.c_str(), w, h, 4, flipped.data(), w*4);
}

void OpenGLRenderer::generateData(std::string_view path, const int count) {
    const auto pixels = m_render_texture.getData();
    constexpr auto size {Settings::Render::render_texture_size};
    saveTexAsImg(data::path(std::format("{}/{:04}.png", path, count)), size, size, pixels);
}

void OpenGLRenderer::initSprites() {
    constexpr int size{Settings::Game::board_size};
    for (int i{0}; i < size; ++i) {
        for (int j{0}; j < size; ++j) {
            m_sprites.emplace_back(
                glm::vec2{j, i},
                "#000000");
        }
    }
}

void OpenGLRenderer::updateSprites(const std::string_view board) {
    size_t idx {0};
    for (const auto c: board) {
        if (c == '\n')
            continue;
        m_sprites[idx++].setColor(color_map[c]);
    }
}

