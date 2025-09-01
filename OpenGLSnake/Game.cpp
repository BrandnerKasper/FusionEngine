#include <print>

#include "Game.h"


Game::Game() {
    Input::setContext(&m_renderer);
}


void Game::run() {
    while (m_running) {
        processInput();
        update();
        render();
    }
}

void Game::processInput() {
    if (Input::pressed(Input::Quit)) {
        m_renderer.close();
        m_running = false;
    }
}

void Game::update() {
    // Update game logic
}


void Game::render() {
    m_renderer.draw();
}
