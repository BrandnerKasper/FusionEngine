#ifndef REDS_H
#define REDS_H

#include <torch/torch.h>

namespace datasets {

    class Reds final : public torch::data::datasets::Dataset<Reds>{
    public:
        explicit Reds(const std::string& dataset_dir, int sequence_length = 3, int crop_size = 0, int scale_factor = 2);
        torch::data::Example<> get(size_t index) override;
        [[nodiscard]] torch::optional<size_t> size() const override;

    private:
        std::string dataset_dir;
        std::vector<std::string> image_paths;
        int sequence_length;
        int crop_size;
        int scale_factor;
    };
}


#endif //REDS_H
