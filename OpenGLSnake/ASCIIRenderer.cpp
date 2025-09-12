#include <iostream>
#include <fstream>
#include <format>

#include "ASCIIRenderer.h"
#include "assets.h"


void ASCIIRenderer::draw(const std::string_view board) {
    create(board);
    std::cout << state << std::endl;
}

void ASCIIRenderer::generateData() const {
    static int count {};

    std::ofstream out(data::path(std::to_string(count++)+".txt"));
    if (!out) {
        throw std::runtime_error("Writing ASCII to txt file failed.");
    }
    out << state;
    out.close();
}

void ASCIIRenderer::create(const std::string_view board) {
    state = {};
    for (const auto c: board) {
        if (c == '\n')
            state += c;
        else {
            state += ascii[c];
        }
    }
}
