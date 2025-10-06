#pragma once
#include <torch/torch.h>

#include "../Renderer/Shader.h"
#include "../Renderer/Mesh.h"


class NeuralTexture {
public:
    NeuralTexture(int size);
    virtual ~NeuralTexture();

    void uploadTensor(const torch::Tensor &t) const;
    void draw();
    void setView(int width, int height);

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
    Shader m_present {"shaders/present.vert", "shaders/present.frag"};
    int m_size;
};