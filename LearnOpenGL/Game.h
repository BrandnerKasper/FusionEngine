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

    // Camera stuff
    glm::vec3 m_cameraPos {glm::vec3(0.0f, 0.0f, 3.0f)};
    glm::vec3 m_cameraFront {glm::vec3(0.0f, 0.0f, -1.0f)};;
    glm::vec3 m_cameraUp {glm::vec3(0.0f, 1.0f, 0.0f)};;

    // Delta Time
    double m_deltaTime {0.0f};
    double m_last_frame {0.0f};

    // More Cubes
    std::array<glm::vec3, 10> m_cubePositions{
        glm::vec3( 0.0f, 0.0f, 0.0f),
        glm::vec3( 2.0f, 5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3( 2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f, 3.0f, -7.5f),
        glm::vec3( 1.3f, -2.0f, -2.5f),
        glm::vec3( 1.5f, 2.0f, -2.5f),
        glm::vec3( 1.5f, 0.2f, -1.5f),
        glm::vec3(-1.3f, 1.0f, -1.5f)
    };
};
