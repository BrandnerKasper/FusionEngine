#include <vector>
#include <iostream>
#include <sstream>

#include "Game.h"
#include "../random.h"


Game::Game(){
    init();
    setPlayer();
    generatePellet();
}

bool Game::run(const double deltaTime, const IInput::Action action) {
    m_lastUpdate += deltaTime;
    if (m_lastUpdate >= Settings::Game::tick) {
        validateAction(action);
        if (m_running)
            update();
        m_lastUpdate -= Settings::Game::tick;
    }
    return m_running;
}

std::string Game::getBoardState() const {
    auto s = m_board.toString();

    std::istringstream iss(s);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(iss, line))
        lines.push_back(line);

    std::reverse(lines.begin(), lines.end());

    s = {};
    for (const auto& l: lines) {
        s += l + "\n";
    }

    return s;
}


void Game::update() {
    m_player.move(m_last_action);

    // Free tail pos so player can move there
    if (const auto tail_pos = m_player.tail_pos; tail_pos.has_value())
        m_board(tail_pos.value()).type = Tile::Empty;

    // Check game state
    bool generate_pellet = false;
    const auto head_pos = m_player.head_pos;
    // Eat Pellet ?
    if (m_board(head_pos).type == Tile::Pellet) {
        m_player.eat();
        generate_pellet = true;
    }
    // Hit Wall?
    else if (m_board(head_pos).type != Tile::Empty)
        m_running = false;

    setPlayer();

    if (generate_pellet)
        generatePellet();
}

void Game::init() {
    constexpr int size {Settings::Game::board_size};
    Tile tile {};
    for (int i {0}; i < size; ++i) {
        for (int j {0}; j < size; ++j) {
            tile.pos = {j, i};
            if (i == 0 || i == size-1 || j == 0 || j == size-1)
                tile.type = Tile::Wall;
            else
                tile.type = Tile::Empty;
            m_board(j, i) = tile;
        }
    }
}

void Game::setPlayer() {
    for (const auto& [pos, type] : m_player.body) {
        m_board(pos).type = type;
    }
}


void Game::generatePellet() {
    std::vector<Position> poss_spawn_pos {};
    for (const auto&[pos, type] : m_board()) {
        if (type == Tile::Empty)
            poss_spawn_pos.push_back(pos);
    }

    const auto random_idx = Random::get<size_t>(0, poss_spawn_pos.size()-1);
    const auto random_pos = poss_spawn_pos[random_idx];
    m_board(random_pos).type = Tile::Pellet;
}

void Game::validateAction(const IInput::Action action) {
    // Don't move into opposite dir
    switch (action) {
        case IInput::Up:
            if (m_last_action == IInput::Down)
                return;
            break;
        case IInput::Left:
            if (m_last_action == IInput::Right)
                return;
            break;
        case IInput::Right:
            if (m_last_action == IInput::Left)
                return;
            break;
        case IInput::Down:
            if (m_last_action == IInput::Up)
                return;
            break;
        default:
            std::cerr << "Validated Input Action that Game can not handle:" << action << std::endl;
            break;
    }
    m_last_action = action;
}

void Game::reset() {
    init();
    m_player.reset();
    setPlayer();
    generatePellet();
    m_running = true;
    m_last_action = IInput::Up;
}
