#pragma once


class Mesh {
public:
    explicit Mesh();
    void draw() const;
    virtual ~Mesh();

private:
    unsigned int m_VAO;
};