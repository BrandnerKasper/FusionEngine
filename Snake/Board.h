#pragma once
#include <array>

#include "settings.h"
#include "types.h"


struct Board {
    explicit Board();

    void draw();
    void generatePellet();
    [[nodiscard]] size_t findIdxOfPos(const Types::Position pos) const {
        return pos.x + Settings::BOARD_SIZE * pos.y;
    }

    std::array<Types::Tile, Settings::BOARD_SIZE * Settings::BOARD_SIZE> body;
};
