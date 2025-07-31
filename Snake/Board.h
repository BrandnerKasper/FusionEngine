#pragma once
#include <array>

#include "settings.h"
#include "types.h"


struct Board {
    explicit Board();

    void draw();
    void generatePellet();

    void setValue(const Types::Position& pos, std::string_view icon);
    std::string getValue(const Types::Position& pos);

    std::array<Types::Tile, Settings::BOARD_SIZE * Settings::BOARD_SIZE> body;
};
