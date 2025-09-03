#include <glad/glad.h>

#include "Mesh.h"


Mesh::Mesh(const std::vector<float> &vertices)
    : m_numberOfVertices(vertices.size()){
    glGenBuffers(1, &m_VBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices.data()), vertices.data(), GL_STATIC_DRAW);
}

void Mesh::draw() const {
    glDrawArrays(GL_TRIANGLES, 0, static_cast<int>(m_numberOfVertices));
}

Mesh::~Mesh() {
    glDeleteBuffers(1, &m_VBO);
}
