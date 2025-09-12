#pragma once
#include <GLFW/glfw3.h>
#include <string_view>
#include <vector>
#include <unordered_map>

#include "../settings.h"
#include "Shader.h"
#include "Sprite.h"
#include "RenderTexture.h"


class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    virtual ~Renderer();
    void draw(std::string_view board);

private:
    void init();
    void initSprites();
    void updateSprites(std::string_view board);

private:
    GLFWwindow* m_window;
    glm::mat4 camera;

    std::vector<Sprite> m_sprites;

    RenderTexture m_render_texture {Settings::Game::board_size * Settings::Render::tile_size};

    std::unordered_map<char, std::string_view> color_map {
        {'0', {"#222222"}},
        {'1', {"#f0f0f0"}},
        {'2', {"#849476"}},
        {'3', {"#b1556c"}},
    };
};