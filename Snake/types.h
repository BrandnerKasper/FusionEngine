#pragma once
#include <string>


namespace Types {
    struct Position {
        int x {};
        int y {};

        Position& operator+=(const Position& other) {
            x += other.x;
            y += other.y;
            return *this;
        }
    };


    struct Tile {
        std::string icon {" "};
        Position pos {};
    };
}