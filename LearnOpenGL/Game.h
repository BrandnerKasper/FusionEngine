#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string_view>

#include "Shader.h"

class Game {
public:
    Game(int width = 800, int height = 600);
    void run();
    virtual ~Game();

private:
    void processInput();

private:
    GLFWwindow* m_window;
    int m_width, m_height;

    std::string_view m_vertex_shader_path {"shaders/vertex.vert"};
    std::string_view m_fragment_shader_path {"shaders/fragment.frag"};
    std::unique_ptr<Shader> m_shader;

    unsigned int m_VBO, m_VAO, m_EBO;

    std::string_view m_texture_path1 {"textures/container.jpg"};
    std::string_view m_texture_path2 {"textures/awesomeface.png"};
    unsigned int texture1, texture2;
    float m_mix {0.2f};
};
