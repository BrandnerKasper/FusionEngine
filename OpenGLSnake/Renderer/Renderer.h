#pragma once
#include <GLFW/glfw3.h>


class Renderer {
public:
    static void draw();

private:
    static inline GLFWwindow* m_window;

    friend class Application;
};