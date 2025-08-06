#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string_view>

#include "Shader.h"

class Game {
public:
    Game();
    void run() const;
    virtual ~Game();

private:
    GLFWwindow* m_window;

    std::string_view m_vertex_shader_path {"shaders/vertex.vert"};
    std::string_view m_fragment_shader_path {"shaders/fragment.frag"};
    std::unique_ptr<Shader> m_shader;

    unsigned int m_VBO, m_VAO, m_EBO;

    std::string_view m_texture_path {"textures/container.jpg"};
};
