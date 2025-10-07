#pragma once
#include <string_view>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "NeuralTexture.h"
#include "../settings.h"


class NeuralRenderer {
public:
    explicit NeuralRenderer(GLFWwindow* window);
    virtual ~NeuralRenderer();
    void draw(std::string_view board);

private:
    void init();
    void initCamera();
    void initNeuralNetwork();

    torch::Tensor inference(const std::string& board);

private:
    GLFWwindow* m_window;
    glm::mat4 m_camera;
    torch::jit::script::Module m_neural_model;
    NeuralTexture m_neural_texture {Settings::Game::board_size * Settings::Render::tile_size};
};

