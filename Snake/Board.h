#pragma once
#include <vector>
#include <string>

#include "Player.h"
#include "types.h"


class Board {
public:
    explicit Board(const std::vector<std::string>& board);
    explicit Board(Player& player, int size);

    void update(Player& player);
    [[nodiscard]] std::vector<Types::Tile> getBody() const { return m_body;}
    [[nodiscard]] int getSize() const { return m_size;}

private:
    void create();
    void generatePellet();
    [[nodiscard]] int findTile(const Types::Position pos) const {
        return pos.x + m_size * pos.y;
    }

private:
    // TODO: use std::array!!
    std::vector<Types::Tile> m_body;
    int m_size;
};
