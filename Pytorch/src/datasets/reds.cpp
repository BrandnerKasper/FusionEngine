#include "reds.h"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <spdlog/fmt/fmt.h>

#include "../utils/utils.h"


datasets::Reds::Reds(const std::string &dataset_dir, int sequence_length, int crop_size, int scale_factor)
    : dataset_dir(dataset_dir), sequence_length(sequence_length), crop_size(crop_size), scale_factor(scale_factor) {
    // Dictionary: sequence -> list of filenames
    std::map<std::string, std::vector<std::string>> sequence_map;

    for (const auto &sequence: std::filesystem::directory_iterator(dataset_dir + "/LR")) {
        // Iterate over files in each sequence
        std::vector<std::string> file_paths{};
        for (const auto &file: std::filesystem::directory_iterator(sequence.path())) {
            file_paths.push_back(file.path().filename().string());
        }
        utils::sort(file_paths);
        for (int i = 0; i < sequence_length; i++) {
            file_paths.erase(file_paths.begin());
        }
        sequence_map[sequence.path().stem()] = file_paths;
    }
    // Flatten the dictionary
    for (const auto &[sequence, files]: sequence_map) {
        for (const auto &file: files) {
            image_paths.push_back(std::filesystem::path(sequence) / file);
        }
    }
}


torch::data::Example<> datasets::Reds::get(size_t index) {
    // LR
    std::vector<torch::Tensor> lr_tensors;
    std::vector<cv::Mat> lr_images;
    // Find the position of '/'
    size_t pos = image_paths[index].find('/');
    // Extract substrings
    std::string sequence = image_paths[index].substr(0, pos);
    std::string file_name = image_paths[index].substr(pos + 1);
    for (int i = 0; i <= sequence_length; i++) {
        std::string lr_frame_path = std::filesystem::path(dataset_dir) / "LR"/ sequence / fmt::format("{:08}.png", std::stoi(file_name) - i);
        cv::Mat lr_img = cv::imread(lr_frame_path, cv::IMREAD_COLOR);
        if (lr_img.empty())
            throw std::runtime_error("Could not open image: " + lr_frame_path);
        lr_images.push_back(lr_img);
    }

    // HR
    std::string hr_frame_path = std::filesystem::path(dataset_dir) / "HR"/ image_paths[index];
    cv::Mat hr_img = cv::imread(hr_frame_path, cv::IMREAD_COLOR);
    if (hr_img.empty())
        throw std::runtime_error("Could not open image: " + hr_frame_path);

    // Random Crop
    if (crop_size != 0)
        utils::vsrRandomCrop(lr_images, hr_img, crop_size, scale_factor);

    // Store LR frames
    // Convert to tensor
    for (const auto &image: lr_images) {
        auto lr_tensor = torch::from_blob(
            image.data, {image.rows, image.cols, 3}, torch::kUInt8).permute({2, 0, 1});
        lr_tensor = lr_tensor.to(torch::kFloat) / 255.0;
        lr_tensors.push_back(lr_tensor);
    }

    // Store HR target frame
    torch::Tensor hr_tensor = torch::from_blob(
        hr_img.data, {hr_img.rows, hr_img.cols, 3}, torch::kUInt8).permute({2, 0, 1});
    hr_tensor = hr_tensor.to(torch::kFloat) / 255.0;

    // return tuple of lr and hr (lr input, hr target)
    return {torch::stack(lr_tensors), hr_tensor};
}


torch::optional<size_t> datasets::Reds::size() const {
    return image_paths.size();
}
