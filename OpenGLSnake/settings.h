#pragma once

#include <string>


namespace Settings {
    namespace Window {
        constexpr float width{512.0f};
        constexpr float height{512.0f};
        constexpr std::string title{"SNAKE"};
    };

    namespace Game {
        constexpr double tick = 0.3;
        constexpr int board_size {32};
        constexpr std::pair start_position {board_size/2-1, board_size/2-1};
        constexpr int body_length {3};
    }

    namespace Render {
        constexpr double frame_time = 1.0f/60.0f;
        constexpr int tile_size {1};
    }
}
