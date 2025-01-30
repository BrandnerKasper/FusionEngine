#include <gtest/gtest.h>

#include "../src/datasets/reds.h"
#include "../src/datasets/urban100.h"
#include "../src/utils/utils.h"


TEST(DataSet, GetImages) {
    const auto dataset = std::make_shared<datasets::Urban100>(std::string(PROJECT_ROOT_DIR) + "/data/Urban100/val");
    for (int i = 0; i < dataset->size(); i++) {
        const auto sample = dataset->get(i);
        auto data = sample.data;
        auto target = sample.target;

        std::cout << "Data: " << data << std::endl;
        std::cout << "Target: " << target << std::endl;
        utils::showTensorAsCVImg(data, target);
    }
}


TEST(REDS, GetFrames) {
    const auto dataset = std::make_shared<datasets::Reds>(std::string(PROJECT_ROOT_DIR) + "/data/REDS/val", 3, 256, 2);
    for (int i = 0; i < dataset->size(); i++) {
        const auto sample = dataset->get(i);
        auto data = sample.data;
        auto target = sample.target;

        std::cout << "Data: " << data.sizes() << std::endl;
        std::cout << "Target: " << target.sizes() << std::endl;
        utils::showVSRData(data, target);
    }
}
