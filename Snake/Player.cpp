#include <iostream>
#include "Player.h"


Player::Player(const Types::Position start_position, const Input::Action action, const int body_length, const std::string& icon)
    : m_icon{icon}, m_action{action} {
    // Create Body of player
    for (int i{0}; i < body_length; ++i) {
        m_body.push_back(Types::Tile(icon, {start_position.x, start_position.y + i}));
    }
}

void Player::move(const Types::Position &dir) {
    prev = m_body[0].pos;
    m_body[0].pos += dir;
    next = m_body[0].pos;
    auto next = prev;
    for (auto i{1}; i < m_body.size(); ++i) {
        prev = m_body[i].pos;
        m_body[i].pos = next.value();
        next = prev;
    }
}

Types::Position getDirectionFromAction(const Input::Action action) {
    Types::Position dir {0, 0};
    switch (action) {
        case Input::Up:
            dir.y -= 1;
            break;
        case Input::Left:
            dir.x -= 1;
            break;
        case Input::Down:
            dir.y += 1;
            break;
        case Input::Right:
            dir.x += 1;
            break;
        case Input::Quit:
            break;
        default:
            std::cerr << "Action not set!" << std::endl;
            break;
    }
    return dir;
}

void Player::setAction(const Input::Action action) {
    // Don't move into opposite dir
    // TODO refactor
    const auto [cur_x, cur_y] = getDirectionFromAction(m_action);
    const auto [new_x, new_y] = getDirectionFromAction(action);
    if (cur_x * -1 == new_x || cur_y * -1 == new_y)
        return;
    m_action = action;
}

void Player::update() {
    const auto dir = getDirectionFromAction(m_action);

    prev = m_body[0].pos;
    m_body[0].pos += dir;
    next = m_body[0].pos;
    auto next = prev;
    for (auto i {1}; i < m_body.size(); ++i) {
        prev = m_body[i].pos;
        m_body[i].pos = next.value();
        next = prev;
    }
}

void Player::eat() {
    m_body.push_back({m_icon, prev.value()});
    prev = std::nullopt;
}
