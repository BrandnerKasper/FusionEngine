#include "urban100.h"

#include <filesystem>
#include <opencv2/opencv.hpp>

#include "../utils/utils.h"


datasets::Urban100::Urban100(const std::string &dataset_dir, const int crop_size, const int scale_factor)
    : crop_size(crop_size), scale_factor(scale_factor) {
    // Init LR paths
    for (const auto &entry: std::filesystem::directory_iterator(dataset_dir + "/LR")) {
        lr_image_paths.push_back(entry.path().string());
    }
    utils::sort(lr_image_paths);

    // Init HR paths
    for (const auto &entry: std::filesystem::directory_iterator(dataset_dir + "/HR")) {
        hr_image_paths.push_back(entry.path().string());
    }
    utils::sort(hr_image_paths);
}

torch::data::Example<> datasets::Urban100::get(const size_t index) {
    // load image via OpenCV
    // LR
    cv::Mat lr_img = cv::imread(lr_image_paths[index], cv::IMREAD_COLOR);
    if (lr_img.empty())
        throw std::runtime_error("Could not open image: " + lr_image_paths[index]);

    // HR
    cv::Mat hr_img = cv::imread(hr_image_paths[index], cv::IMREAD_COLOR);
    if (hr_img.empty())
        throw std::runtime_error("Could not open image: " + hr_image_paths[index]);

    // Random Crop
    if (crop_size != 0)
        utils::randomCrop(lr_img, hr_img, crop_size, scale_factor);

    // Convert to tensor
    auto lr_tensor = torch::from_blob(
        lr_img.data, {lr_img.rows, lr_img.cols, 3}, torch::kUInt8).permute({2, 0, 1});
    lr_tensor = lr_tensor.to(torch::kFloat) / 255.0;
    auto hr_tensor = torch::from_blob(
        hr_img.data, {hr_img.rows, hr_img.cols, 3}, torch::kUInt8).permute({2, 0, 1});
    hr_tensor = hr_tensor.to(torch::kFloat) / 255.0;

    // return tuple of lr and hr (lr input, hr target)
    return {lr_tensor, hr_tensor};
}

torch::optional<size_t> datasets::Urban100::size() const {
    return lr_image_paths.size(); // lr and hr have same size
}
