#include <gtest/gtest.h>

#include "../src/datasets/urban100.h"
#include "../src/utils/utils.h"


TEST(DataSet, GetImages) {
    auto datasets = std::make_shared<datasets::Urban100>(std::string(PROJECT_ROOT_DIR) + "/data/Urban100/val");
    for (int i = 0; i < datasets->size(); i++) {
        auto sample = datasets->get(i);
        auto data = sample.data;
        auto target = sample.target;

        std::cout << "Data: " << data << std::endl;
        std::cout << "Target: " << target << std::endl;
        utils::showTensorAsCVImg(data, target);
    }
}
