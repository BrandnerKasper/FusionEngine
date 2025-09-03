#pragma once

#include <string>


namespace Settings {
    namespace Window {
        constexpr int width{1280};
        constexpr int height{720};
        constexpr std::string title{"SNAKE"};
    };

    namespace Game {
        constexpr int board_size {10};
        constexpr std::pair start_position {6, 2};
        constexpr int body_length {3};
    }
}
