#pragma once
#include <vector>


class Mesh {
public:
    explicit Mesh(const std::vector<float>& vertices);
    void draw() const;
    virtual ~Mesh();

private:
    unsigned int m_VBO = 0;
    unsigned int m_numberOfVertices;
};