#pragma once
#include <vector>
#include <optional>

#include "Tile.h"
#include "Game.h"


class Player {
public:
    Player();

    void move(Game::Action action);
    void eat();

    std::vector<Tile> body {};
    std::optional<Position> tail {std::nullopt};
private:
    void init();
};