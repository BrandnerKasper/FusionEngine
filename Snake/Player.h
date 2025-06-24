#pragma once
#include <vector>
#include <string>
#include <optional>

#include "input.h"
#include "types.h"


class Player {
public:
    explicit Player(Types::Position start_position = Types::Position{6, 2}, Input::Action action = Input::Up,
                    int body_length = 3, const std::string &icon = "■");

    [[nodiscard]] std::vector<Types::Tile> getBody() const { return m_body;}

    void setAction(Input::Action action);

    void move(Input::Action action);

    void eat();

    [[nodiscard]] std::optional<Types::Position> getPrev() const {return prev;}
    [[nodiscard]] std::optional<Types::Position> getNext() const {return next;}

private:
    std::string m_icon {};
    std::vector<Types::Tile> m_body {};
    std::optional<Types::Position> prev {std::nullopt};
    std::optional<Types::Position> next {std::nullopt};
    Input::Action m_action {};
};
