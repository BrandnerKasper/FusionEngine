#include "Player.h"
#include "../settings.h"


Player::Player() {
    init();
}

void Player::init() {
    for (int i {0}; i < Settings::Game::body_length; ++i) {
        body.push_back(Tile{{Settings::Game::start_position.first, Settings::Game::start_position.second - i}, Tile::Player});
    }
}

void Player::move(const Input::Action action) {
    auto head = body[0];
    tail_pos = head.pos;

    switch (action) {
        case Input::Up:
            head.pos.y += 1;
            break;
        case Input::Left:
            head.pos.x -= 1;
            break;
        case Input::Down:
            head.pos.y -= 1;
            break;
        case Input::Right:
            head.pos.x += 1;
            break;
        default:
            break;
    }
    head_pos = head.pos;
    body[0].pos = head_pos;

    for (int i {1}; i < body.size(); ++i) {
        auto temp = body[i].pos;
        body[i].pos = tail_pos.value();
        tail_pos = temp;
    }
}

void Player::eat() {
    body.push_back({tail_pos.value(), Tile::Player});
    tail_pos = std::nullopt;
}

void Player::reset() {
    body.clear();
    init();
}
