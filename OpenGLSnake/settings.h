#pragma once

#include <string>


namespace Settings {
    namespace Window {
        constexpr int width{1280};
        constexpr int height{720};
        constexpr std::string title{"SNAKE"};
    };

    namespace Game {
        constexpr double tick = 1.2;
        constexpr int board_size {10};
        constexpr std::pair start_position {4, 4};
        constexpr int body_length {3};
    }

    namespace Render {
        constexpr double frame_time = 0.2;
    }
}
