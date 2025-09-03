#include "Player.h"
#include "../settings.h"


Player::Player() {
    init();
}

void Player::init() {
    for (int i {0}; i < Settings::Game::body_length; ++i) {
        body.push_back(Tile{{Settings::Game::start_position.first, Settings::Game::start_position.second + i}, Tile::Player});
    }
}

void Player::move(const Game::Action action) {
    auto head = body[0];
    tail = head.pos;

    switch (action) {
        case Game::Up:
            head.pos.y -= 1;
            break;
        case Game::Left:
            head.pos.x -= 1;
            break;
        case Game::Down:
            head.pos.y += 1;
            break;
        case Game::Right:
            head.pos.x += 1;
            break;
        default:
            break;
    }

    for (int i {1}; i < body.size(); ++i) {
        auto temp = body[i].pos;
        body[i].pos = tail.value();
        tail = temp;
    }
}

void Player::eat() {
    body.push_back({tail.value(), Tile::Player});
    tail = std::nullopt;
}
