#include <glad/glad.h>

#include "Mesh.h"

// float vertices[] = {
//     // pos      // tex
//     0.0f, 1.0f, 0.0f, 1.0f,
//     1.0f, 0.0f, 1.0f, 0.0f,
//     0.0f, 0.0f, 0.0f, 0.0f,
//
//     0.0f, 1.0f, 0.0f, 1.0f,
//     1.0f, 1.0f, 1.0f, 1.0f,
//     1.0f, 0.0f, 1.0f, 0.0f
// };


Mesh::Mesh(const std::vector<float>& vertices) {
    unsigned int VBO;
    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(m_VAO);
    // Vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<long>(vertices.size() * sizeof(float)), vertices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0); // pos
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1); // uv
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), reinterpret_cast<void *>(2 * sizeof(float)));
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void Mesh::draw() const {
    glBindVertexArray(m_VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
}

Mesh::~Mesh() {
    glDeleteVertexArrays(1, &m_VAO);
}
