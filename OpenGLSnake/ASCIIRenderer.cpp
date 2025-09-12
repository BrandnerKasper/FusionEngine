#include <iostream>

#include "ASCIIRenderer.h"


void ASCIIRenderer::draw(const std::string_view board) {
    create(board);
    std::cout << state << std::endl;
}

void ASCIIRenderer::create(const std::string_view board) {
    for (const auto c: board) {
        if (c == '\n')
            state += c;
        else {
            state += ascii[c];
        }
    }
}
