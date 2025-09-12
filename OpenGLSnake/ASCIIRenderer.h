#pragma once
#include <unordered_map>
#include <string>
#include <string_view>


class ASCIIRenderer {
public:
    ASCIIRenderer() = default;

    void draw(std::string_view board);
    void generateData() const;

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