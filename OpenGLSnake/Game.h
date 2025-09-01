#pragma once
#include <GLFW/glfw3.h>


class Game {
public:
    explicit Game(int width = 800, int height = 600);
    ~Game();

    void run();

private:
    bool update();
    void render();
    void processInput();

    void initWindow();

private:
    GLFWwindow* m_window;
    int m_width, m_height;
};