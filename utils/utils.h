#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <opencv2/opencv.hpp>


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

}

#endif //UTILS_H
