#pragma once
#include <GLFW/glfw3.h>
#include <string_view>
#include <vector>
#include <unordered_map>

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
    void createBuffer();

private:
    GLFWwindow* m_window;

    Shader m_shader {"shaders/vertex.vert", "shaders/fragment.frag"};
    std::vector<Sprite> m_sprites;


    unsigned int m_FBO {0};
    RenderTexture m_render_texture {};

    std::unordered_map<char, std::string_view> color_map {
        {'0', {"#222222"}},
        {'1', {"#f0f0f0"}},
        {'2', {"#849476"}},
        {'3', {"#b1556c"}},
    };
};