#ifndef SISR_H
#define SISR_H
#include <torch/torch.h>

namespace models {
    struct SISR : torch::nn::Module {
        // Variables
        // Layers
        torch::nn::Conv2d conv1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::Conv2d conv3{nullptr};
        // Scaling
        int _scale_factor;

        explicit SISR(int scale_factor = 2);

        // Functions
        torch::Tensor forward(torch::Tensor x);
        void measureInference(torch::Tensor &input);
    };
    void testSISR();

} // models

#endif //SISR_H
