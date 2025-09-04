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
    void initSprites();
    void updateSprites(std::string_view board);
private:
    GLFWwindow* m_window;

    // glm::mat4 camera = glm::ortho(0.0f, 800.0f, 600.0f, 0.0f);
    // std::string_view m_vertex_shader_path {"shaders/vertex.vert"};
    // std::string_view m_fragment_shader_path {"shaders/fragment.frag"};
    Shader m_shader {"shaders/vertex.vert", "shaders/fragment.frag"};
    std::vector<Sprite> m_sprites;
    unsigned int quadVAO;

    std::unordered_map<char, glm::vec3> color_map {
        {'0', {0, 0, 0}},
        {'1', {1, 0, 0}},
        {'2', {0, 1, 0}},
        {'3', {0, 0, 1}},
    };
};
