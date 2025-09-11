#pragma once

#include "Mesh.h"
#include "Shader.h"


class RenderTexture {
public:
    RenderTexture();

    void draw();
    unsigned int m_colorTex {0};

private:
    void create();

private:
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
    int m_offW {32}, m_offH {32}; // size of board
    Shader m_present {"shaders/present.vert", "shaders/present.frag"};
};