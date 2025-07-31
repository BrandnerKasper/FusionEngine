#pragma once
#include <vector>
#include <string>
#include <optional>

#include "input.h"
#include "types.h"


struct Player {
    explicit Player(Types::Position start_position = Types::Position{6, 2}, Input::Action action = Input::Up,
                    int body_length = 3, const std::string &icon = "■");

    void move(Input::Action action);
    void eat();

    std::string icon {};
    std::vector<Types::Tile> body {};
    std::optional<Types::Position> head {std::nullopt};
    std::optional<Types::Position> tail {std::nullopt};
    Input::Action last_action {};
};
