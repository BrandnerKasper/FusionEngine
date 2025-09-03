#include <vector>
#include <iostream>

#include "Game.h"
#include "../random.h"


Game::Game(){
    init();
    setPlayer();
    generatePellet();
}

void Game::update(double deltaTime, Input::Action action) {
    validateAction(static_cast<Action>(action));
    m_player.move(m_last_action);

    // Free tail pos so player can move there
    const auto tail = m_player.tail;
    if (tail.has_value())
        setTile(tail.value(), Tile::Empty);

    // Check game state
    bool generate_pellet = false;
    const auto head_pos = m_player.body[0].pos;
    // Eat Pellet ?
    if (getTile(head_pos).type == Tile::Pellet) {
        m_player.eat();
        generate_pellet = true;
    }
    // Hit Wall?
    else if (getTile(head_pos).type != Tile::Empty)
        m_running = false;

    if (generate_pellet)
        generatePellet();
}

void Game::init() {
    constexpr int size {Settings::Game::board_size};
    Tile tile {};
    for (int i {0}; i < size; ++i) {
        for (int j {0}; j < size; ++j) {
            tile.pos = {j, i};
            if (i == 0 || j == 0)
                tile.type = Tile::Wall;
            else
                tile.type = Tile::Empty;
            m_board[i * size + j] = tile;
        }
    }
}

void Game::setPlayer() {
    for (const auto& [pos, type] : m_player.body) {
        setTile(pos, type);
    }
}

void Game::setTile(const Position pos, const Tile::Type type) {
    const size_t idx = pos.x + Settings::Game::board_size * pos.y;
    m_board[idx].type = type;
}

Tile& Game::getTile(Position pos) {
    const size_t idx = pos.x + Settings::Game::board_size * pos.y;
    return m_board[idx];
}

void Game::generatePellet() {
    std::vector<size_t> poss_spawn_pos {};
    for (size_t i {0}; i < m_board.size(); ++i) {
        if (m_board[i].type == Tile::Empty)
            poss_spawn_pos.push_back(i);
    }

    const auto random_idx = Random::get<size_t>(0, poss_spawn_pos.size()-1);
    const auto random_pos = poss_spawn_pos[random_idx];
    m_board[random_pos].type = Tile::Pellet;
}

void Game::validateAction(const Action action) {
    // Don't move into opposite dir
    switch (action) {
        case Up:
            if (m_last_action == Down)
                return;
            break;
        case Left:
            if (m_last_action == Right)
                return;
            break;
        case Right:
            if (m_last_action == Left)
                return;
            break;
        case Down:
            if (m_last_action == Up)
                return;
            break;
        default:
            std::cerr << "Should not come to this!" << std::endl;
            break;
    }
    m_last_action = action;
}
