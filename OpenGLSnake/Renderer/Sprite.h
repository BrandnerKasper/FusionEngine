#pragma once
#include <string_view>

#include "Mesh.h"
#include "Shader.h"


class Sprite {
public:
    Sprite(const Shader& shader, glm::vec2 position, std::string_view hex_color);

    void draw() const;
    void setColor(std::string_view hex_color);

private:
    Mesh m_mesh {};
    Shader m_shader;

    glm::vec2 m_position;
    glm::vec3 m_color;
    float m_size = 16.0f;
};
