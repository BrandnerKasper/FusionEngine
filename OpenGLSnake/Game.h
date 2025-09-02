#pragma once


class Game {
public:
    Game() = default;
    ~Game() = default;

    void update(double deltaTime);

private:
    // actual Game
    bool m_running {true};
};