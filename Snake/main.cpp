#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include "input.h"
#include "types.h"
#include "Player.h"
#include "Board.h"

#pragma region Game

// TODO REFACTOR THIS
#define START_POSITION Position {6, 2}

constexpr auto TICK = std::chrono::milliseconds(1200);
constexpr auto FRAME_TIME = std::chrono::milliseconds(200);
using CLOCK = std::chrono::steady_clock;

//TODO
bool GAME = true;

void update(Player& player, const Input::Action action, Board& board) {

    switch (action) {
        case Input::Quit:
            GAME = false;
            return;
        default:
            break;
    }

    player.setAction(action);
    player.update();
    board.update(player);
}

void render(const Board& board) {
    int length {};
    for (auto& [icon, pos]: board.getBody()) {
        if (icon == "┐") length = pos.x + 1;
    }

    int counter = {0};
    for (const auto& [icon, pos] : board.getBody()) {
        std::cout << icon;
        ++counter;
        if (counter % length == 0)
            std::cout << std::endl;
    }
}

void sleep(const int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


#pragma endregion


int main() {
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
    int size {10};
    auto player {Player{{size/2, size/2}}};
    Board board {player, size};

    Input::setNonBlockingInput(true);

    auto lastUpdate = CLOCK::now();;
    auto lastRender = CLOCK::now();;
    auto input = Input::Up;

    while (GAME) {
        // Time Management
        auto currentTime = CLOCK::now();

        // Input
        Input::inputPooling(input);

        // Logic
        if (currentTime - lastUpdate >= TICK) {
            update(player, input, board);
            lastUpdate += TICK;
        }

        // Render
        if (currentTime-lastRender >= FRAME_TIME) {
            render(board);
            lastRender += FRAME_TIME;
        }
    }

    return 0;
}
