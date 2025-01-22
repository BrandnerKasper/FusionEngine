#include <gtest/gtest.h>

#include "../models/sisr.h"


TEST(SISRTest, StressTest) {
    // Initialize the model
    models::SISR model;
    auto dummy_input = torch::rand({1, 3, 1024, 1024});
    // Perform the stress test
    std::cout << "Stress Testing SISR with " << dummy_input.sizes() << "\n";
    model.measureInference(dummy_input);
}