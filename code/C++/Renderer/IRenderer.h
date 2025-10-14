#pragma once

#include <string_view>


struct IRenderer {
    virtual ~IRenderer() = default;

    virtual void draw(std::string_view board) = 0;
    virtual void generateData(std::string_view path, int count) {};
};