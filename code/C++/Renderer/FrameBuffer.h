#pragma once

#include "Mesh.h"
#include "Shader.h"


class FrameBuffer {
public:
    FrameBuffer(int size, std::string vp, std::string fp);
    virtual ~FrameBuffer();

    void draw() const;

    void bind() const;
    void unbind() const;
    void setData(const unsigned char* data, int width, int height) const;
    const std::vector<unsigned char>& getData();

private:
    void create();
    void destroy();

private:
    unsigned int m_FBO {};
    unsigned int m_Tex {};

    Mesh m_ndc{
            {
                -1.f, -1.f, 0.f, 0.f,
                1.f, -1.f, 1.f, 0.f,
                1.f, 1.f, 1.f, 1.f,
                -1.f, -1.f, 0.f, 0.f,
                1.f, 1.f, 1.f, 1.f,
                -1.f, 1.f, 0.f, 1.f
            }
    };
    int m_size;
    Shader m_shader;
    std::vector<unsigned char> m_data;
};