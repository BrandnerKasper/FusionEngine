#pragma once
#include <GLFW/glfw3.h>
#include <string_view>
#include <vector>
#include <unordered_map>

#include "FrameBuffer.h"
#include "IRenderer.h"
#include "../settings.h"
#include "Shader.h"
#include "Sprite.h"


class OpenGLRenderer final : public IRenderer {
public:
    explicit OpenGLRenderer(GLFWwindow* window);
    ~OpenGLRenderer() override;

    void draw(std::string_view board) override;
    void generateData(std::string_view path, int count) override;

private:
    void init();
    void initCamera();
    void initSprites();
    void updateSprites(std::string_view board);

private:
    GLFWwindow* m_window;
    glm::mat4 camera;

    std::vector<Sprite> m_sprites;

    int m_size {Settings::Game::board_size * Settings::Render::tile_size};
    FrameBuffer m_render_texture {m_size, "shaders/present.vert", "shaders/present.frag"};

    std::unordered_map<char, std::string_view> color_map {
        {'0', {"#222222"}},
        {'1', {"#f0f0f0"}},
        {'2', {"#849476"}},
        {'3', {"#b1556c"}},
    };
};