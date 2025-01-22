#include "sisr.h"
#include <chrono>

models::SISR::SISR(const int scale_factor) {
    // Define the layers
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 9).stride(1).padding(4)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1).padding(0)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 3, 5).stride(1).padding(2)));
    // Scaling
    _scale_factor = scale_factor;
}

torch::Tensor models::SISR::forward(torch::Tensor x) {
    x = torch::nn::functional::interpolate(
            x,
            torch::nn::functional::InterpolateFuncOptions()
            .scale_factor(std::vector<double>{2.0, 2.0})
            .mode(torch::kBilinear)
            .align_corners(false));
    x = torch::relu(conv1->forward(x));
    x = torch::relu(conv2->forward(x));
    x = conv3->forward(x);
    return x;
}

void models::SISR::measureInference(torch::Tensor &input) {
    std::cout << "---------------------------------\n";
    // Check if CUDA is available
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA for stress test.\n";
    } else {
        std::cout << "Using CPU for stress test.\n";
    }

    // Move the model and input to the device
    this->to(device);
    this->eval(); // Set the model to evaluation mode
    input = input.to(device);

    // Perform warm-up iterations (for CUDA)
    for (int i = 0; i < 10; i++) {
        auto output = this->forward(input);
    }

    // Measure inference time over multiple iterations
    constexpr int iterations{100};
    std::chrono::duration<double> elapsed{0};
    for (int i = 0; i < iterations; i++) {
        if (device.is_cuda()) {
            torch::cuda::synchronize(); // Ensure all GPU operations are done
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        auto output = this->forward(input);
        if (device.is_cuda()) {
            torch::cuda::synchronize(); // Ensure all GPU operations are done
        }
        auto endTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsedTime = endTime - startTime;
        elapsed += elapsedTime;
    }
    elapsed /= iterations;

    // Log the results
    std::cout << "Average inference time: " << elapsed.count() * 1000.0f << " milliseconds\n";
    std::cout << "---------------------------------\n";
}
