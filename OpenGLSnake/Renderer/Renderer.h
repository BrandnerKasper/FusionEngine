#pragma once
#include <GLFW/glfw3.h>
#include <string_view>
#include <vector>
#include <unordered_map>

#include "Shader.h"
#include "Sprite.h"


class Renderer {
public:
    Renderer(GLFWwindow* window);
    virtual ~Renderer();
    void draw(std::string_view board);

private:
    void init();
    void initSprites();
    void updateSprites(std::string_view board);
    void createTex();

private:
    GLFWwindow* m_window;

    // glm::mat4 camera = glm::ortho(0.0f, 800.0f, 600.0f, 0.0f);
    // std::string_view m_vertex_shader_path {"shaders/vertex.vert"};
    // std::string_view m_fragment_shader_path {"shaders/fragment.frag"};
    Shader m_shader {"shaders/vertex.vert", "shaders/fragment.frag"};
    std::vector<Sprite> m_sprites;

    Mesh m_ndc{
        {
            -1.f, -1.f, 0.f, 0.f,
             1.f, -1.f, 1.f, 0.f,
             1.f,  1.f, 1.f, 1.f,
            -1.f, -1.f, 0.f, 0.f,
             1.f,  1.f, 1.f, 1.f,
            -1.f,  1.f, 0.f, 1.f
        }
    };
    unsigned int m_FBO, m_colorTex;
    int m_offW {32}, m_offH {32};
    Shader m_present {"shaders/present.vert", "shaders/present.frag"};


    std::unordered_map<char, std::string_view> color_map {
        {'0', {"222222"}},
        {'1', {"f0f0f0"}},
        {'2', {"849476"}},
        {'3', {"b1556c"}},
    };
};