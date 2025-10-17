#pragma once

#include <string>
#include <unordered_map>


namespace Settings {
    // TODO: add keys for Input bindings;

    namespace Window {
        constexpr int width{512};
        constexpr int height{512};
        constexpr std::string title{"SNAKE"};
    };

    namespace Game {
        constexpr double tick = 0.1;
        constexpr int board_size {32};
        constexpr int body_length {3};
    }

    namespace Render {
        constexpr std::string curr_Renderer {"Neural"};
        constexpr double frame_time = 1.0f/60.0f;
        constexpr int tile_size {1};
        constexpr int render_texture_size {Game::board_size * tile_size};
    }

    namespace Data {
        constexpr bool generate{false};
        constexpr int amount {1000};
    }
}
