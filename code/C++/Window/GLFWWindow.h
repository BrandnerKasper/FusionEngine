#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "IWindow.h"
#include "../settings.h"


class GLFWWindow {
public:
    GLFWWindow();
    virtual ~GLFWWindow();

    [[nodiscard]] GLFWwindow* get() const {return m_window;}
    void setTitle(const std::string& t) const;
    bool shouldClose() const;

private:
    void init();

private:
    // Window
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};
};