#pragma once
#include <vector>


class Mesh {
public:
    explicit Mesh(const std::vector<float>& vertices);
    void draw() const;
    virtual ~Mesh();

private:
    unsigned int m_VAO {0};
};