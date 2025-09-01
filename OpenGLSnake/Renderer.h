#pragma once
#include <GLFW/glfw3.h>
#include <string>

#include "settings.h"

class Renderer {
public:
    explicit Renderer();
    virtual ~Renderer();

    void draw();
    void close();

    [[nodiscard]] GLFWwindow* getWindow() const {return m_window;}

private:
    void init();

private:
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};
};