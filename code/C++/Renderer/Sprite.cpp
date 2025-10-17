#include <sstream>
#include <iomanip>

#include "Sprite.h"

glm::vec3 hexToRGBVec(std::string_view hex_color) {
    if (!hex_color.empty() && hex_color.front() == '#')
        hex_color.remove_prefix(1); // drop the '#'

    unsigned int rgb = 0;
    std::stringstream ss;
    ss << std::hex << hex_color;
    ss >> rgb;

    float r = ((rgb >> 16) & 0xFF) / 255.0f;
    float g = ((rgb >> 8)  & 0xFF) / 255.0f;
    float b = ((rgb)       & 0xFF) / 255.0f;

    return {r, g, b};
}

Sprite::Sprite(const glm::vec2 position, const std::string_view hex_color)
    : m_position{position}, m_color{hexToRGBVec(hex_color)}{}

void Sprite::draw(const glm::mat4& projection) const {
    m_shader.use();

    auto model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(m_position.x*m_size, m_position.y*m_size, 0.0f));
    model = glm::scale(model, glm::vec3(m_size, m_size, 1.0f));

    // Could give the shader a model view projection matrix
    m_shader.setValue("model", model);
    m_shader.setValue("color", m_color);
    m_shader.setValue("projection", projection);

    m_mesh.draw();
}

void Sprite::setColor(const std::string_view hex_color) {
    m_color = hexToRGBVec(hex_color);
}


