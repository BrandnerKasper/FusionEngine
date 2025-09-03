#pragma once
#include <array>
#include <vector>

#include "../settings.h"
#include "../Input.h"
#include "Tile.h"
#include "Player.h"


struct Board {
    Board(const int r, const int c)
        : rows{r}, cols{c}, data(r*c) {}

    int rows {};
    int cols {};
    std::vector<Tile> data;

    Tile& operator()(const Position& pos) {return data[pos.x + rows * pos.y];}
};

class Game {
public:
    enum Action {
        Up, Down, Left, Right,
    };

    explicit Game();
    virtual ~Game() = default;

    void update(double deltaTime, Input::Action action);

private:
    void init();
    void setPlayer();
    void setTile(Position pos, Tile::Type type);
    Tile& getTile(Position pos);
    void generatePellet();
    void validateAction(Action action);

private:
    bool m_running {true};

    // TODO use Grid class instead of std::array -> write get and set methods based on pos
    std::array<Tile, Settings::Game::board_size * Settings::Game::board_size> m_board;
    // Board m_board {Settings::Game::board_size, Settings::Game::board_size};
    Action m_last_action {Up};
    Player m_player {};
};