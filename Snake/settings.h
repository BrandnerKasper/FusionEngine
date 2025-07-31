#pragma once
#include <chrono>

namespace Settings {
    constexpr auto TICK = std::chrono::milliseconds(1200);
    constexpr auto FRAME_TIME = std::chrono::milliseconds(200);
    constexpr auto BOARD_SIZE = 10;
}