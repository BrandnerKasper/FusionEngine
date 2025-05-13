#ifndef METRICS_H
#define METRICS_H

#include <torch/torch.h>

namespace metrics {

    struct Metrics {
    public:
        Metrics(const torch::Tensor &img1, const torch::Tensor &img2);
        Metrics(float _psnr, float _ssim);
        Metrics &operator+=(const Metrics& other);
        Metrics &operator/=(const float divisor);

        float psnr;
        float ssim;
    };
}

#endif //METRICS_H
