#include "Sprite.h"

Sprite::Sprite(const Shader& shader, const glm::vec2 position, const glm::vec3 color)
    : m_shader(shader), m_position{position}, m_color{color}{}

void Sprite::draw() const {
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(m_position.x*m_size, m_position.y*m_size, 0.0f));


    model = glm::scale(model, glm::vec3(m_size, m_size, 1.0f));

    m_shader.setValue("model", model);
    m_shader.setValue("color", m_color);

    m_mesh.draw();
}


