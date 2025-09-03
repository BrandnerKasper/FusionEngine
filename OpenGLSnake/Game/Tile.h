#pragma once

struct Position {
    int x{};
    int y{};

    Position &operator+=(const Position &other) {
        x += other.x;
        y += other.y;
        return *this;
    }
};

struct Tile {

    enum Type {
        Empty,
        Wall,
        Player,
        Pellet
    };

    Position pos {};
    Type type {};
};