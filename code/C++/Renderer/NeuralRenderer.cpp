#include <glm/ext/matrix_clip_space.hpp>
#include  <vector>

#include "NeuralRenderer.h"
#include "../settings.h"
#include "../assets.h"

// UTILITY
torch::Tensor asciiToOneHotTensor(const std::string& ascii) {
    std::istringstream iss(ascii);
    std::string line;
    std::vector<std::string> lines;

    while (std::getline(iss, line))
        lines.push_back(std::move(line));

    std::vector<int64_t> data;
    data.reserve(32*32);

    for (const auto& l: lines)
        for (const auto& c : l)
            data.push_back(c-'0');

    auto t = torch::tensor(data, torch::dtype(torch::kLong)).view({32, 32});

    // one hot encode
    int K = 4;
    t = torch::one_hot(t, K).permute({2, 0, 1}).to(torch::kFloat32);

    return t;
}


NeuralRenderer::NeuralRenderer(GLFWwindow* window)
    : m_window{window}{
    init();
}

void NeuralRenderer::init() {
    initCamera();
    initNeuralNetwork();
}

void NeuralRenderer::initCamera() {
    constexpr auto size = static_cast<float>(Settings::Game::board_size);
    m_camera = glm::ortho(0.0f, size, size, 0.0f, -1.0f, 1.0f);
}

void NeuralRenderer::initNeuralNetwork() {
    m_neural_model = torch::jit::load(assets::path("pretrained_models/PNN_W.pt"));
    m_neural_model.eval();
    m_neural_model.to(torch::kCPU);
}

NeuralRenderer::~NeuralRenderer() {
    m_window = nullptr;
}

void NeuralRenderer::draw(const std::string_view board) {
    auto pred = inference(std::string(board));
    m_neural_texture.setData(pred.data_ptr<unsigned char>(), pred.size(0), pred.size(1));

    glDisable(GL_DEPTH_TEST);

    int winW, winH;
    glfwGetFramebufferSize(m_window, &winW, &winH);
    glViewport(0, 0, winW, winH);
    m_neural_texture.unbind();
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    m_neural_texture.draw();

    glfwSwapBuffers(m_window);
    glfwPollEvents();
}

torch::Tensor NeuralRenderer::inference(const std::string &board) {
    const auto one_hot = asciiToOneHotTensor(std::string(board));

    torch::NoGradGuard ng;
    std::vector<torch::jit::IValue> inputs {one_hot};
    auto pred = m_neural_model.forward(inputs).toTensor();
    pred = pred.clamp(0, 1).mul(255.0).to(torch::kU8).permute({1,2,0}).contiguous();
    return pred;
}