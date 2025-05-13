#include "datasets/reds.h"
#include "datasets/urban100.h"
#include "models/sisr.h"
#include "utils/utils.h"
#include "metrics/metrics.h"


void train() {
    // track logged info in separate file
    utils::setupLogger();
    // Net
    const auto net = std::make_shared<models::SISR>();
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
            const auto input = batch.data.to(torch::kCUDA);
            const auto target = batch.target.to(torch::kCUDA);

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
            metrics::Metrics val_metrics {0.0, 0.0};
            size_t count = 0;
            for (auto &batch: *val_dataloader) {
                const auto input = batch.data.to(torch::kCUDA);
                const auto target = batch.target.to(torch::kCUDA);

                const auto prediction = net->forward(input);

                val_metrics += metrics::Metrics(prediction, target);
                count++;
            }
            val_metrics /= count;
            spdlog::warn("Validation | PSNR: {} | SSIM: {}", val_metrics.psnr, val_metrics.ssim);
        }

        torch::save(net, std::string(PROJECT_ROOT_DIR) + "/net.pt");
    }
}


int main() {
    train();
    return 0;
}
