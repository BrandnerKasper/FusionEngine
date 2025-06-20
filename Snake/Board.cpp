#include "Board.h"


#include <iostream>

#include "random.h"


std::vector<std::string> splitUTF8Chars(const std::string& input) {
    std::vector<std::string> result;
    size_t i = 0;

    while (i < input.size()) {
        unsigned char c = input[i];
        size_t char_len = 1;

        if ((c & 0x80) == 0x00) char_len = 1;         // 1-byte (ASCII)
        else if ((c & 0xE0) == 0xC0) char_len = 2;    // 2-byte
        else if ((c & 0xF0) == 0xE0) char_len = 3;    // 3-byte
        else if ((c & 0xF8) == 0xF0) char_len = 4;    // 4-byte
        else throw std::runtime_error("Invalid UTF-8 encoding");

        result.push_back(input.substr(i, char_len));
        i += char_len;
    }
}

Board::Board(const std::vector<std::string>& board) {
    m_size = static_cast<int>(board[0].size());
    for (auto i {0}; i < board.size(); ++i) {
        std::vector<std::string> line = splitUTF8Chars(board[i]);
        for (auto j {0}; j < line.size(); ++j)
            m_body.push_back({line[j], {j, i}});
    }
}

Board::Board(Player& player, const int size = 10)
    : m_size(size){
    create();

    // TODO: This could be done by the game class -> we should take the ref of the player out here
    for (const auto& [icon, pos] : player.getBody()) {
        m_body[findTile(pos)].icon = icon;
    }

    generatePellet();
}

void Board::update(Player &player) {
    // Update prev first so tail pos is free for player to move
    if (const auto prev = player.getPrev(); prev.has_value())
        m_body[findTile(prev.value())].icon = " ";

    bool generate_pellet = false;
    if (const auto next = player.getNext(); next.has_value()) {
        // Pellet
        if (m_body[findTile(next.value())].icon == "▫") {
            player.eat();
            generate_pellet = true;
        }
        // Wall
        else if (m_body[findTile(next.value())].icon != " ")
            auto a = false;
            // TODO let game handle this logic! GAME = false;
        // TODO Bug 1: snake should only be able to move into 3 directions
    }

    for (const auto& [icon, pos] : player.getBody()) {
        m_body[findTile(pos)].icon = icon;
    }

    if (generate_pellet)
        generatePellet();
}


void Board::create() {
    std::string icon{};
    for (int i{0}; i < m_size; ++i) {
        for (int j{0}; j < m_size; ++j) {
            icon = " ";
            Types::Position pos{j, i};
            if (i == 0) {
                icon = "─";
                if (j == 0) icon = "┌";
                if (j == m_size - 1) icon = "┐";
            } else if (i == m_size - 1) {
                icon = "─";
                if (j == 0) icon = "└";
                if (j == m_size - 1) icon = "┘";
            } else if (j == 0 || j == m_size - 1) {
                icon = "│";
            }
            m_body.push_back({icon, {j, i}});
        }
    }
}

void Board::generatePellet() {
    std::vector<size_t> poss_spawn_pos {};
    for (size_t i {0}; i < m_body.size(); ++i) {
        if (m_body[i].icon == " ")
            poss_spawn_pos.push_back(i);
    }
    const auto random_idx = Random::get<size_t>(0, (poss_spawn_pos.size()-1));
    const auto random_pos = poss_spawn_pos[random_idx];
    m_body[random_pos].icon = "▫";
}


