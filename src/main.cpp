#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "../datasets/urban100.h"
#include "../models/sisr.h"
#include "../utils/utils.h"


float calculatePSNR(const torch::Tensor &prediction, const torch::Tensor &target) {
    constexpr float max_val = 1.0f; //our tensors are normalized in range [0, 1]
    // Ensure the images are the same size
    if (!prediction.sizes().equals(target.sizes())) {
        throw std::invalid_argument("Input tensors must have the same dimensions.");
    }

    // Compute Mean Squared Error (MSE)
    auto mse = torch::mean(torch::pow(prediction - target, 2)).item<float>();

    // Avoid division by zero
    if (mse == 0.0) {
        return std::numeric_limits<float>::infinity(); // Infinite PSNR for identical images
    }

    // Calculate PSNR
    return 20.0 * std::log10(max_val) - 10.0 * std::log10(mse);
}


// Turn RGB tensor into greyscale tensor
torch::Tensor rgb_to_grayscale(const torch::Tensor &img) {
    // Ensure input shape is (C, H, W)
    TORCH_CHECK(img.dim() == 3 && img.size(0) == 3, "Expected (3, H, W) tensor");

    // Grayscale conversion weights (matching OpenCV and PIL)
    const torch::Tensor weights = torch::tensor({0.2989, 0.5870, 0.1140}, img.options());

    // Apply grayscale conversion and add batch dimension
    return (img * weights.view({3, 1, 1})).sum(0).unsqueeze(0); // Result is (1, H, W)
}


// Gaussian kernel function
torch::Tensor createGaussianKernel(int kernel_size, float sigma) {
    torch::Tensor kernel_1d = torch::arange(kernel_size).to(torch::kFloat) - (kernel_size - 1) / 2.0;
    kernel_1d = torch::exp(-0.5 * torch::pow(kernel_1d / sigma, 2));
    kernel_1d = kernel_1d / kernel_1d.sum();

    torch::Tensor kernel_2d = torch::mm(kernel_1d.view({kernel_size, 1}), kernel_1d.view({1, kernel_size}));
    return kernel_2d / kernel_2d.sum();
}

// Apply Gaussian blur
torch::Tensor gaussianBlur(torch::Tensor img, int kernel_size, float sigma) {
    auto kernel = createGaussianKernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0);
    kernel = kernel.expand({img.size(1), 1, kernel_size, kernel_size}).to(img.device());

    return torch::conv2d(img, kernel, {}, 1, (kernel_size - 1) / 2);
}


// we calc SSIM only for greyscale images
float calculateSSIM(const torch::Tensor &img1, const torch::Tensor &img2, const int kernel_size = 11,
                    const float sigma = 1.5) {
    constexpr float C1 = 0.01 * 0.01;
    constexpr float C2 = 0.03 * 0.03;

    const torch::Tensor mu1 = gaussianBlur(img1, kernel_size, sigma);
    const torch::Tensor mu2 = gaussianBlur(img2, kernel_size, sigma);

    const torch::Tensor mu1_sq = mu1.pow(2);
    const torch::Tensor mu2_sq = mu2.pow(2);
    const torch::Tensor mu1_mu2 = mu1 * mu2;

    const torch::Tensor sigma1_sq = gaussianBlur(img1 * img1, kernel_size, sigma) - mu1_sq;
    const torch::Tensor sigma2_sq = gaussianBlur(img2 * img2, kernel_size, sigma) - mu2_sq;
    const torch::Tensor sigma12 = gaussianBlur(img1 * img2, kernel_size, sigma) - mu1_mu2;

    const torch::Tensor ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                                       (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2));

    return ssim_map.mean().item<float>();
}


void train() {
    // track logged info in separate file
    utils::setupLogger();
    // Net
    auto net = std::make_shared<models::SISR>();
    net->to(torch::kCUDA);

    // Data
    // Train
    auto train_dataset = datasets::Urban100(std::string(PROJECT_ROOT_DIR) + "/data/Urban100/train", 128)
            .map(torch::data::transforms::Stack<>());
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), 16);
    // Validation
    auto val_dataset = datasets::Urban100(std::string(PROJECT_ROOT_DIR) + "/data/Urban100/val")
            .map(torch::data::transforms::Stack<>());
    auto val_dataloader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), 1);

    // Hyperparameter
    constexpr float learning_rate = 0.01;
    auto optimizer = torch::optim::Adam(net->parameters(), learning_rate);
    constexpr int max_epoch = 100;

    spdlog::info("Start training for {}", max_epoch);
    // Loop
    for (size_t epoch = 0; epoch < max_epoch; epoch++) {
        float epoch_loss = 0.0;
        size_t batch_count = 0;

        // train
        for (auto &batch: *train_dataloader) {
            auto input = batch.data.to(torch::kCUDA);
            auto target = batch.target.to(torch::kCUDA);

            optimizer.zero_grad();
            auto prediction = net->forward(input);
            auto loss = torch::l1_loss(prediction, target);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<float>();
            batch_count++;
        }
        float avg_loss = epoch_loss / batch_count;
        spdlog::info("Epoch: {} | Loss: {:6f}", epoch, avg_loss);

        // val
        if ((epoch + 1) % 10 == 0) {
            float psnr = 0.0;
            float ssim = 0.0;
            size_t count = 0;
            for (auto &batch: *val_dataloader) {
                auto input = batch.data.to(torch::kCUDA);
                auto target = batch.target.to(torch::kCUDA);

                auto prediction = net->forward(input);

                psnr += calculatePSNR(prediction, target);
                auto img1 = rgb_to_grayscale(prediction.squeeze(0)).to(torch::kCUDA);
                auto img2 = rgb_to_grayscale(target.squeeze(0)).to(torch::kCUDA);
                ssim += calculateSSIM(img1, img2);
                count++;
            }
            const float avg_psnr = psnr / count;
            const float avg_ssim = ssim / count;
            spdlog::warn("Validation | PSNR: {} | SSIM: {}", avg_psnr, avg_ssim);
        }

        torch::save(net, std::string(PROJECT_ROOT_DIR) + "/net.pt");
    }
}


int main() {
    train();
    return 0;
}
