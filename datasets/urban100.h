#ifndef URBAN100_H
#define URBAN100_H

#include <torch/torch.h>


namespace datasets {

    class Urban100 final : public torch::data::datasets::Dataset<Urban100> {

    public:
        explicit Urban100(const std::string &dataset_dir, int crop_size = 0, int scale_factor = 2);
        torch::data::Example<> get(size_t index) override;
        [[nodiscard]] torch::optional<size_t> size() const override;

    private:
        std::vector<std::string> lr_image_paths;
        std::vector<std::string> hr_image_paths;
        int crop_size;
        int scale_factor;
    };
}

#endif //URBAN100_H
