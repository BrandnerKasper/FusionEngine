#pragma once

#include <string>


namespace Settings {
    namespace Window {
        constexpr float width{512.0f};
        constexpr float height{512.0f};
        constexpr std::string title{"SNAKE"};
    };

    namespace Game {
        constexpr double tick = 1.2;
        constexpr int board_size {8};
        constexpr std::pair start_position {3, 3};
        constexpr int body_length {3};
    }

    namespace Render {
        constexpr double frame_time = 0.2;
    }
}
