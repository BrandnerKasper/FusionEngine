#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

#include "IWindow.h"
#include "../settings.h"


class GLFWWindow final : public IWindow {
public:
    GLFWWindow();

    ~GLFWWindow() override;

    [[nodiscard]] void* get() const override {return m_window;}
    void setTitle(const std::string& t) const override;
    bool shouldClose() const override;

private:
    void init();

private:
    GLFWwindow* m_window;
    int m_width {Settings::Window::width};
    int m_height {Settings::Window::height};
    std::string m_title{Settings::Window::title};
};