#pragma once
#include <vector>

#include "../settings.h"
#include "../Input.h"
#include "Tile.h"
#include "Player.h"


struct Board {
    explicit Board(const int r_c)
        : rows_cols{r_c}, data(r_c*r_c) {}

    int rows_cols {};
    std::vector<Tile> data;

    Tile& operator()(const Position& pos) {return data[pos.x + rows_cols * pos.y];}
    Tile& operator()(const int x, const int y) {return data[x + rows_cols * y];}
    [[nodiscard]] size_t size() const {return data.size();}

    std::vector<Tile>& operator()() {
        return data;
    }
    const std::vector<Tile>& operator()() const {
        return data;
    }

    [[nodiscard]] std::string toString() const {
        std::string str {};
        int counter {0};
        for (auto [pos, type]: data) {
            str += std::to_string(type);
            ++counter;
            if (counter % Settings::Game::board_size == 0)
                str += "\n";
        }
        return str;
    }
};

class Game {
public:
    explicit Game();
    virtual ~Game() = default;

    bool run(double deltaTime, Input::Action action);

    std::string getBoardState() const;

    void reset();

private:
    void init();
    void update();
    void setPlayer();
    void generatePellet();
    void validateAction(Input::Action action);

private:
    bool m_running {true};
    double m_lastUpdate{};

    Board m_board {Settings::Game::board_size};
    Input::Action m_last_action {Input::Up};
    Player m_player {};
};