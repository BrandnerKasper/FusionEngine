#pragma once
#include <chrono>

#include "Board.h"
#include "Player.h"


class Game {
public:
    explicit Game();
    ~Game();

    void run();

private:
    bool update();
    void render();

private:
    Player m_player;
    Board m_board;
    Input::Action m_input;
    bool running;

    std::chrono::steady_clock::time_point m_lastUpdate;
    std::chrono::steady_clock::time_point m_lastRender;

};

/* Example Board
        ┌────────────┐
        │            │
        │     ■      │
        │     ■  ▫   │
        │     ■      │
        │            │
        │            │
        │            │
        │            │
        │            │
        │            │
        │            │
        │            │
        └────────────┘
*/