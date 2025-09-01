#pragma once
#include <string>

#include "Renderer.h"
#include "Input.h"


class Game {
public:
    Game();
    ~Game() = default;

    void run();

private:
    void processInput();
    void update();
    void render();

private:
    // actual Game
    bool m_running {true};

    // Rendering
    Renderer m_renderer;
    Input m_input;
    int m_width, m_height;
};