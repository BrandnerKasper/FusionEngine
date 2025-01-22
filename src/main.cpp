#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "../datasets/urban100.h"

// Setup logging function
void setupLogger() {
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

        // Ensure macro expansion and proper path concatenation
        std::string log_file_path = std::string(PROJECT_ROOT_DIR) + "/logs/log.txt";

        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, true);

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("train_logger", sinks.begin(), sinks.end());

        logger->set_level(spdlog::level::info); // Set log level
        logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v"); // Timestamp + level + message

        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger); // Set as default logger

        logger->info("Logger initialized. Log file at: {}", log_file_path);
    } catch (const spdlog::spdlog_ex &ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
    }
}


// MNIST end to end example
struct Net : torch::nn::Module {
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, 0.5, is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), 1);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


void MNISTEndToEnd() {
    // Our defined struct network
    auto net = std::make_shared<Net>();
    net->to(torch::kCUDA);

    // multi thread dataloader (true!)io
    const auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST(std::string(PROJECT_ROOT_DIR) + "/data/MNIST").map(
            torch::data::transforms::Stack<>()),
        64);

    // Optimizer
    constexpr float learning_rate = 0.01;
    torch::optim::SGD optimizer(net->parameters(), learning_rate);

    // Train Loop
    for (size_t epoch = 0; epoch <= 1000; ++epoch) {
        size_t batch_index = 0;

        for (auto &batch: *data_loader) {
            auto data = batch.data.to(torch::kCUDA);
            auto target = batch.target.to(torch::kCUDA);
            optimizer.zero_grad();
            torch::Tensor prediction = net->forward(data);
            torch::Tensor loss = torch::nll_loss(prediction, target);
            loss.backward();
            optimizer.step();

            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << std::endl;
                torch::save(net, std::string(PROJECT_ROOT_DIR) + "/net.pt");
            }
        }
    }
}


// OpenCV read image example
void readImageOpenCV() {
    const std::string image_path = std::string(PROJECT_ROOT_DIR) + "/data/test.png";
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "Could not open image: " << image_path << std::endl;
    }

    // Convert image to a tensor (H, W, C -> C, H, W)
    auto tensor = torch::from_blob(
        image.data, {image.rows, image.cols, 3}, torch::kUInt8).permute({2, 0, 1});

    // Normalize image to [0, 1] range
    tensor = tensor.to(torch::kFloat) / 255.0;

    std::cout << tensor << std::endl;

    cv::imshow("test", image);
    cv::waitKey(0);
}


void showTensorAsCVImg(torch::Tensor lr_tensor, torch::Tensor hr_tensor) {
    // Ensure tensors are contiguous
    lr_tensor = lr_tensor.permute({1, 2, 0}).contiguous();
    hr_tensor = hr_tensor.permute({1, 2, 0}).contiguous();

    // Process LR tensor
    lr_tensor = lr_tensor.mul(255).clamp(0, 255).to(torch::kByte);
    int lr_channels = lr_tensor.size(2); // Number of channels
    cv::Mat lr_img;
    if (lr_channels == 1) {
        lr_img = cv::Mat(lr_tensor.size(0), lr_tensor.size(1), CV_8UC1, lr_tensor.data_ptr());
    } else if (lr_channels == 3) {
        lr_img = cv::Mat(lr_tensor.size(0), lr_tensor.size(1), CV_8UC3, lr_tensor.data_ptr());
    } else {
        std::cerr << "Unsupported number of channels in LR tensor: " << lr_channels << std::endl;
        return;
    }

    // Process HR tensor
    hr_tensor = hr_tensor.mul(255).clamp(0, 255).to(torch::kByte);
    int hr_channels = hr_tensor.size(2); // Number of channels
    cv::Mat hr_img;
    if (hr_channels == 1) {
        hr_img = cv::Mat(hr_tensor.size(0), hr_tensor.size(1), CV_8UC1, hr_tensor.data_ptr());
    } else if (hr_channels == 3) {
        hr_img = cv::Mat(hr_tensor.size(0), hr_tensor.size(1), CV_8UC3, hr_tensor.data_ptr());
    } else {
        std::cerr << "Unsupported number of channels in HR tensor: " << hr_channels << std::endl;
        return;
    }

    // Ensure dimensions match for concatenation
    if (lr_img.rows != hr_img.rows || lr_img.cols != hr_img.cols) {
        cv::resize(lr_img, lr_img, hr_img.size()); // Resize LR image to match HR image
    }

    // Ensure types match for concatenation
    if (lr_img.type() != hr_img.type()) {
        if (hr_img.type() == CV_8UC3 && lr_img.channels() == 1) {
            cv::cvtColor(lr_img, lr_img, cv::COLOR_GRAY2BGR); // Convert grayscale to RGB
        }
    }

    // Combine LR and HR images for comparison
    cv::Mat combined;
    cv::hconcat(lr_img, hr_img, combined); // Combine LR and HR horizontally

    // Display the images
    cv::imshow("LR and HR Images", combined);
    cv::waitKey(0);
    cv::destroyAllWindows();
}



void testUrban100Dataset() {
    auto datasets = std::make_shared<datasets::Urban100>(std::string(PROJECT_ROOT_DIR) + "/data/Urban100/val");
    for (int i = 0; i < datasets->size(); i++) {
        auto sample = datasets->get(i);
        auto data = sample.data;
        auto target = sample.target;

        std::cout << "Data: " << data << std::endl;
        std::cout << "Target: " << target << std::endl;
        showTensorAsCVImg(data, target);
    }
}


struct SimpleSISRNet : torch::nn::Module {
    // Layers
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::Conv2d conv3{nullptr};

    SimpleSISRNet() {
        // Define the layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 9).stride(1).padding(4)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1).padding(0)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 3, 5).stride(1).padding(2)));
    }

    torch::Tensor forward(torch::Tensor x) {
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
};


void testSimpleSISRNet() {
    auto net = std::make_shared<SimpleSISRNet>();
    net->to(torch::kCUDA);
    auto dummy_input = torch::rand({1, 3, 512, 512});
    dummy_input = dummy_input.to(torch::kCUDA);
    auto test = net->forward(dummy_input);
    std::cout << test.sizes() << std::endl;
}


void measureInferenceOfModel(SimpleSISRNet &model, torch::Tensor &input) {
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
    model.to(device);
    model.eval(); // Set the model to evaluation mode
    input = input.to(device);

    // Perform warm-up iterations (for CUDA)
    for (int i = 0; i < 10; i++) {
        auto output = model.forward(input);
    }

    // Measure inference time over multiple iterations
    constexpr int iterations{100};
    std::chrono::duration<double> elapsed{0};
    for (int i = 0; i < iterations; i++) {
        if (device.is_cuda()) {
            torch::cuda::synchronize(); // Ensure all GPU operations are done
        }
        auto startTime = std::chrono::high_resolution_clock::now();
        auto output = model.forward(input);
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


void testMeasure() {
    // Initialize the model
    SimpleSISRNet model;
    auto dummy_input = torch::rand({1, 3, 1024, 1024});
    // Perform the stress test
    measureInferenceOfModel(model, dummy_input); // Set `true` to use CUDA if available
}


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
    setupLogger();
    // Net
    auto net = std::make_shared<SimpleSISRNet>();
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
    // testUrban100Dataset();
    return 0;
}
