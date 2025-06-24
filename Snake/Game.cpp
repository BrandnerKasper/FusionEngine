#include "Game.h"

#include "settings.h"


Game::Game()
    : m_player{Player{{Settings::BOARD_SIZE / 2, Settings::BOARD_SIZE / 2}}}, m_board{}, running{true}
{
    Input::setNonBlockingInput(true);
    m_lastUpdate = std::chrono::steady_clock::now();
    m_lastRender = std::chrono::steady_clock::now();

    // Start with Up movement
    m_input = Input::Up;
    // Set player in board
    for (const auto& [icon, pos] : m_player.getBody()) {
        m_board.body[m_board.findIdxOfPos(pos)].icon = icon;
    }
    // Generate first pellet
    m_board.generatePellet();
}

Game::~Game() {
    Input::setNonBlockingInput(false);
}

void Game::run() {
    while (running) {
        auto currentTime = std::chrono::steady_clock::now();
        Input::inputPooling(m_input);
        if (currentTime - m_lastUpdate >= Settings::TICK) {
            running = update();
            m_lastUpdate += Settings::TICK;
        }
        if (currentTime - m_lastRender >= Settings::FRAME_TIME) {
            render();
            m_lastRender += Settings::FRAME_TIME;
        }
    }
}

bool Game::update() {
    if (m_input == Input::Quit)
        return false;

    m_player.move(m_input);

    // Update prev first so tail pos is free for player to move
    if (const auto prev = m_player.getPrev(); prev.has_value())
        m_board.body[m_board.findIdxOfPos(prev.value())].icon = " ";

    bool generate_pellet = false;
    bool hitWall = false;
    if (const auto next = m_player.getNext(); next.has_value()) {
        // Pellet
        if (m_board.body[m_board.findIdxOfPos(next.value())].icon == "▫") {
            m_player.eat();
            generate_pellet = true;
        }
        // Wall
        else if (m_board.body[m_board.findIdxOfPos(next.value())].icon != " ")
            hitWall = true;
    }

    for (const auto& [icon, pos] : m_player.getBody()) {
        m_board.body[m_board.findIdxOfPos(pos)].icon = icon;
    }

    if (generate_pellet)
        m_board.generatePellet();

    if (hitWall)
        return false;
    return true;
}

void Game::render() {
    m_board.draw();
}