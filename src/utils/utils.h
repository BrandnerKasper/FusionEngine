#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>


namespace utils {
    inline std::mt19937 RANDOM(42); // better Random.h class at: https://www.learncpp.com/cpp-tutorial/global-random-numbers-random-h/

    inline void sort(std::vector<std::string> &paths) {
        // Sort the file paths based on filenames (lexicographically)
        std::sort(paths.begin(), paths.end(),
                  [](const std::string &a, const std::string &b) {
                      return std::filesystem::path(a).filename() < std::filesystem::path(b).filename();
                  });
    }

    inline void randomCrop(cv::Mat &lr_img, cv::Mat &hr_img, const int crop_size, const int scale_factor) {
        // Random top-left corner
        const int x = static_cast<int>(utils::RANDOM() % (lr_img.cols - crop_size + 1));
        const int y = static_cast<int>(utils::RANDOM() % (lr_img.rows - crop_size + 1));

        // Crop LR image in-place
        lr_img = lr_img(cv::Rect(x, y, crop_size, crop_size));

        // Crop HR image in-place
        hr_img = hr_img(
            cv::Rect(x * scale_factor, y * scale_factor, crop_size * scale_factor, crop_size * scale_factor));

        if (!lr_img.isContinuous()) {
            lr_img = lr_img.clone();
        }
        if (!hr_img.isContinuous()) {
            hr_img = hr_img.clone();
        }
    }

    inline void showTensorAsCVImg(torch::Tensor lr_tensor, torch::Tensor hr_tensor) {
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

    inline void setupLogger() {
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
}

#endif //UTILS_H
