#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include <torch/torch.h>
#include <torch/script.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "extern/stb_image_write.h"


std::string ex{
    R"(11111111111111111111111111111111
10000000000003000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000020000000000000001
10000000000000020000000000000001
10000000000000020000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
10000000000000000000000000000001
11111111111111111111111111111111)"
};

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
            data.push_back(static_cast<int64_t>(c-'0'));

    auto t = torch::tensor(data, torch::dtype(torch::kLong)).view({32, 32});

    // one hot encode
    int K = 4;
    t = torch::one_hot(t, K).permute({2, 0, 1}).to(torch::kFloat32);

    return t;
}

int main() {

    std::cout << ex << std::endl;

    // Load example string as torch tensor with one hot encoding
    auto t = asciiToOneHotTensor(ex);

    std::cout << t << std::endl;

    // make a single inference of the model
    torch::jit::script::Module model = torch::jit::load("pretrained_models/PNN_W.pt");
    model.eval();
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    if (torch::cuda::is_available())
        std::cout << "CUDA available!" << std::endl;
    else
        std::cout << "CPU ONLY!" << std::endl;

    model.to(device);
    t = t.to(device);

    std::vector<torch::jit::IValue> inputs {t};
    auto pred = model.forward(inputs).toTensor().to(torch::kCPU);

    auto pred_cpu = pred.clamp(0, 1).mul(255.0).to(torch::kU8).permute({1,2,0}).contiguous();
    // pred_cpu = pred_cpu.flip(0).contiguous();
    const uint8_t* pixels = pred_cpu.data_ptr<uint8_t>();

    // transform and save the result as a .png
    stbi_write_png("out.png", 32, 32, 3, pixels, 32*3);

    return 0;
}