#pragma once
#include <vector>
#include <optional>

#include "Tile.h"
#include "../Input.h"


class Player {
public:
    Player();

    void move(Input::Action action);
    void eat();

    std::vector<Tile> body {};
    Position head_pos{};
    std::optional<Position> tail_pos {std::nullopt};
private:
    void init();
};