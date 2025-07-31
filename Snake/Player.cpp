#include <iostream>
#include "Player.h"


Player::Player(const Types::Position start_position, const Input::Action action, const int body_length, const std::string& icon)
    : icon{icon}, last_action{action} {
    // Create Body of player
    for (int i{0}; i < body_length; ++i) {
        body.push_back(Types::Tile(icon, {start_position.x, start_position.y + i}));
    }
}

void setAction(Player& player, const Input::Action action) {
    // Don't move into opposite dir
    switch (action) {
        case Input::Up:
            if (player.last_action == Input::Down)
                return;
            break;
        case Input::Left:
            if (player.last_action == Input::Right)
                return;
            break;
        case Input::Right:
            if (player.last_action == Input::Left)
                return;
            break;
        case Input::Down:
            if (player.last_action == Input::Up)
                return;
            break;
        default:
            std::cerr << "Should not come to this!" << std::endl;
            break;
    }
    player.last_action = action;
}

void Player::move(const Input::Action action) {
    setAction(*this, action);
    const auto dir = Input::getDirectionFromAction(last_action);

    tail = body[0].pos;
    body[0].pos += dir;
    head = body[0].pos;
    auto temp = tail;
    for (auto i {1}; i < body.size(); ++i) {
        tail = body[i].pos;
        body[i].pos = temp.value();
        temp = tail;
    }
}

void Player::eat() {
    body.push_back({icon, tail.value()});
    tail = std::nullopt;
}
