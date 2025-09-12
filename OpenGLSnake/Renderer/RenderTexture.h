#pragma once

#include "Mesh.h"
#include "Shader.h"


class RenderTexture {
public:
    explicit RenderTexture(int size);
    virtual ~RenderTexture();

    void begin();
    void draw();
    void end(int width, int height);

    void getTextureImage();

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
                 1.f,  1.f, 1.f, 1.f,
                -1.f, -1.f, 0.f, 0.f,
                 1.f,  1.f, 1.f, 1.f,
                -1.f,  1.f, 0.f, 1.f
            }
    };
    int m_size; // size of board
    Shader m_present {"shaders/present.vert", "shaders/present.frag"};
    std::vector<unsigned char> pixels;
};