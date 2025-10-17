#pragma once
#include <string_view>
#include <glm/glm.hpp>

#include "Mesh.h"
#include "Shader.h"
#include "../settings.h"


class Sprite {
public:
    Sprite(glm::vec2 position, std::string_view hex_color);

    void draw(const glm::mat4& projection) const;
    void setColor(std::string_view hex_color);

private:
    Shader m_shader {"shaders/vertex.vert", "shaders/fragment.frag"};
    Mesh m_mesh{
        {
            0.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,

            0.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 0.0f, 1.0f, 0.0f
        }
    };

    glm::vec2 m_position;
    glm::vec3 m_color;
    float m_size {Settings::Render::tile_size};
};
