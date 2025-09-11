#pragma once
#include <string_view>

#include "Mesh.h"
#include "Shader.h"
#include "../settings.h"


class Sprite {
public:
    Sprite(Shader& shader, const std::vector<float>& vertices, glm::vec2 position, std::string_view hex_color);

    void draw() const;
    void setColor(std::string_view hex_color);

private:
    Shader& m_shader;
    Mesh m_mesh;

    glm::vec2 m_position;
    glm::vec3 m_color;
    float m_size {Settings::Render::tile_size};
};
