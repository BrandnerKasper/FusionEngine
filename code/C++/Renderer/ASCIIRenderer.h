#pragma once
#include <unordered_map>
#include <string>

#include "IRenderer.h"


class ASCIIRenderer final : public IRenderer {
public:
    ASCIIRenderer() = default;

    ~ASCIIRenderer() override = default;

    void draw(std::string_view board) override;
    void generateData(std::string_view path, int count) override;

private:
    void create(std::string_view board);

private:
    static inline std::unordered_map<char, std::string> ascii {
                    {'0', " "},
                    {'1', "#"},
                    {'2', "o"},
                    {'3', "•"}
    };
    std::string state {};
};
