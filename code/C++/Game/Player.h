#pragma once
#include <vector>
#include <optional>

#include "Tile.h"
#include "../Input/IInput.h"


class Player {
public:
    Player();

    void move(IInput::Action action);
    void eat();
    void reset();

    std::vector<Tile> body {};
    Position head_pos{};
    std::optional<Position> tail_pos {std::nullopt};
private:
    void init();
};