#include "metrics.h"


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


// Turn RGB tensor into greyscale tensor
torch::Tensor rgb_to_grayscale(const torch::Tensor &img) {
    // Ensure input shape is (C, H, W)
    TORCH_CHECK(img.dim() == 3 && img.size(0) == 3, "Expected (3, H, W) tensor");

    // Grayscale conversion weights (matching OpenCV and PIL)
    const torch::Tensor weights = torch::tensor({0.2989, 0.5870, 0.1140}, img.options());

    // Apply grayscale conversion and add batch dimension
    return (img * weights.view({3, 1, 1})).sum(0).unsqueeze(0); // Result is (1, H, W)
}



metrics::Metrics::Metrics(const torch::Tensor &img1, const torch::Tensor &img2) {
    psnr = calculatePSNR(img1, img2);
    auto pre = rgb_to_grayscale(img1.squeeze(0)).to(torch::kCUDA);
    auto tar = rgb_to_grayscale(img2.squeeze(0)).to(torch::kCUDA);
    ssim = calculateSSIM(pre, tar);
}

metrics::Metrics::Metrics(float _psnr, float _ssim) {
    psnr = _psnr;
    ssim = _ssim;
}

metrics::Metrics & metrics::Metrics::operator+=(const Metrics &other) {
    psnr += other.psnr;
    ssim += other.ssim;
    return *this;
}

metrics::Metrics & metrics::Metrics::operator/=(const float divisor) {
    psnr /= divisor;
    ssim /= divisor;
    return *this;
}
