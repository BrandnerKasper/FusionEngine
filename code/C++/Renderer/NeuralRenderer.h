#pragma once
#include <string_view>
#include  <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <torch/script.h>

#include "../settings.h"
#include "FrameBuffer.h"
#include "IRenderer.h"


class NeuralRenderer final : public IRenderer {
public:
    explicit NeuralRenderer(GLFWwindow* window);
    ~NeuralRenderer() override;
    void draw(std::string_view board) override;

private:
    void init();
    void initCamera();
    void initNeuralNetwork();

    torch::Tensor inference(const std::string& board);

private:
    GLFWwindow* m_window;
    glm::mat4 m_camera;
    torch::jit::script::Module m_neural_model;
    FrameBuffer m_neural_texture {Settings::Game::board_size * Settings::Render::tile_size, "shaders/neural.vert", "shaders/neural.frag"};
};

