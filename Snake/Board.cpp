#include "Board.h"

#include <iostream>

#include "random.h"
#include "settings.h"


Board::Board() {
    // Create Board
    std::string icon{};
    constexpr int size {Settings::BOARD_SIZE};
    for (int i{0}; i < size; ++i) {
        for (int j{0}; j < size; ++j) {
            icon = " ";
            const Types::Position pos{j, i};
            if (i == 0) {
                icon = "─";
                if (j == 0) icon = "┌";
                if (j == size - 1) icon = "┐";
            } else if (i == size - 1) {
                icon = "─";
                if (j == 0) icon = "└";
                if (j == size - 1) icon = "┘";
            } else if (j == 0 || j == size - 1) {
                icon = "│";
            }
            body[i * Settings::BOARD_SIZE + j] = {icon, pos};
        }
    }
}

void Board::draw() {
    int counter = {0};
    for (const auto& [icon, pos] : body) {
        std::cout << icon;
        ++counter;
        if (counter % Settings::BOARD_SIZE == 0)
            std::cout << std::endl;
    }
}

void Board::generatePellet() {
    std::vector<size_t> poss_spawn_pos {};
    for (size_t i {0}; i < body.size(); ++i) {
        if (body[i].icon == " ")
            poss_spawn_pos.push_back(i);
    }
    const auto random_idx = Random::get<size_t>(0, (poss_spawn_pos.size()-1));
    const auto random_pos = poss_spawn_pos[random_idx];
    body[random_pos].icon = "▫";
}


