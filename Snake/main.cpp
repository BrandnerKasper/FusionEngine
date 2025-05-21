#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

#include "random.h"

#pragma region INPUT
void setNonBlockingInput(bool enable) {
    static struct termios oldt, newt;

    if (enable) {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);  // Turn off buffering and echo
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);

        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);  // Non-blocking read
    } else {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);  // Restore terminal
    }
}


enum class Action {
    Up,
    Left,
    Down,
    Right,
    Quit,
    Default
};


Action inputPooling() {
    if (const char ch = getchar(); ch != EOF) {
        switch (ch) {
            case 'w': case 'W':
                std::cout << "UP\n";
                return Action::Up;
            case 'a': case 'A':
                std::cout << "LEFT\n";
                return Action::Left;
            case 's': case 'S':
                std::cout << "DOWN\n";
                return Action::Down;
            case 'd': case 'D':
                std::cout << "RIGHT\n";
                return Action::Right;
            case 'q': case 'Q':
                std::cout << "QUIT\n";
                return Action::Quit;
            default:
                std::cout << "DEFAULT\n";
                return Action::Default;
        }
    }
    return Action::Default;
}
#pragma endregion

#pragma region Game

#define TICK 800
#define START_POSITION Position {1, 2}


bool GAME = true;

struct Position {
    int x {};
    int y {};

    Position& operator+=(const Position& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
};


struct Tile {
    std::string icon {" "};
    Position pos {};
};

Position getDirectionFromAction(const Action action) {
    Position dir {0, 0};
    switch (action) {
        case Action::Up:
            dir.y -= 1;
            break;
        case Action::Left:
            dir.x -= 1;
            break;
        case Action::Down:
            dir.y += 1;
            break;
        case Action::Right:
            dir.x += 1;
            break;
        default:
            std::cerr << std::format("Action not set!");
            break;
    }
    return dir;
}

class Player {
public:
    explicit Player(const std::string& icon = "■", const Position start_position = START_POSITION, const int body_length = 3)
        : m_icon {icon}
    {
        // Create Body of player
        for (int i {0}; i < body_length; ++i) {
            m_body.push_back(Tile(icon, {start_position.x, start_position.y + i}));
        }
    }

    [[nodiscard]] std::vector<Tile> getBody() const { return m_body;}

    void move(const Position& dir) {
        if (dir.x == 0 && dir.y == 0) {
            next = std::nullopt;
            prev = std::nullopt;
            return;
        }
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

    void setAction(const Action action) {
        // Default action should not overwrite our last movement action
        if (action == Action::Default)
            return;
        // Don't move into opposite dir
        const auto [cur_x, cur_y] = getDirectionFromAction(m_action);
        const auto [new_x, new_y] = getDirectionFromAction(action);
        if (cur_x * -1 == new_x || cur_y * -1 == new_y)
            return;
        m_action = action;
    }

    void update() {
        Position dir = getDirectionFromAction(m_action);

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

    void eat() {
        m_body.push_back({m_icon, prev.value()});
        prev = std::nullopt;
    }

    [[nodiscard]] std::optional<Position> getPrev() const {return prev;}
    [[nodiscard]] std::optional<Position> getNext() const {return next;}

private:
    std::string m_icon {};
    std::vector<Tile> m_body {};
    std::optional<Position> prev {std::nullopt};
    std::optional<Position> next {std::nullopt};
    Action m_action {Action::Right};
};

class Board {
public:
    // Based on ASCII art
    explicit Board(const std::vector<std::string>& board) {
        m_size = static_cast<int>(board[0].size());
        for (auto i {0}; i < board.size(); ++i) {
            std::vector<std::string> line = splitUTF8Chars(board[i]);
            for (auto j {0}; j < line.size(); ++j)
                m_body.push_back({line[j], {j, i}});
        }
    }

    // Based on size (quadratic)
    explicit Board(const Player& player, const int size = 10)
        : m_size(size) {
        create();

        // This could be done by the game class -> we should take the ref of the player out here
        for (const auto& [icon, pos] : player.getBody()) {
            m_body[findTile(pos)].icon = icon;
        }

        generatePellet();
    }

    void update(Player& player) {
        // Update prev first so tail pos is free for player to move
        if (const auto prev = player.getPrev(); prev.has_value())
            m_body[findTile(prev.value())].icon = " ";

        bool generate_pellet = false;
        if (const auto next = player.getNext(); next.has_value()) {
            // Pellet
            if (m_body[findTile(next.value())].icon == "▫") {
                player.eat();
                generate_pellet = true;
            }
            // Wall
            else if (m_body[findTile(next.value())].icon != " ")
                GAME = false;
            // TODO Bug 1: snake should only be able to move into 3 directions
        }

        for (const auto& [icon, pos] : player.getBody()) {
            m_body[findTile(pos)].icon = icon;
        }

        if (generate_pellet)
            generatePellet();
    }

    [[nodiscard]] std::vector<Tile> getBody() const { return m_body;}
    [[nodiscard]] int getSize() const { return m_size;}


private:
    void create() {
        std::string icon {};
        for (int i{0}; i< m_size; ++i) {
            for (int j{0}; j < m_size; ++j) {
                icon = " ";
                Position pos {j, i};
                if (i == 0) {
                    icon = "─";
                    if (j == 0) icon = "┌";
                    if (j == m_size-1) icon = "┐";
                }
                else if (i == m_size-1) {
                    icon = "─";
                    if (j == 0) icon = "└";
                    if (j == m_size-1) icon = "┘";
                }
                else if (j == 0 || j == m_size-1) {
                    icon = "│";
                }
                m_body.push_back({icon, {j, i}});
            }
        }
    }

    static std::vector<std::string> splitUTF8Chars(const std::string& input) {
        std::vector<std::string> result;
        size_t i = 0;

        while (i < input.size()) {
            unsigned char c = input[i];
            size_t char_len = 1;

            if ((c & 0x80) == 0x00) char_len = 1;         // 1-byte (ASCII)
            else if ((c & 0xE0) == 0xC0) char_len = 2;    // 2-byte
            else if ((c & 0xF0) == 0xE0) char_len = 3;    // 3-byte
            else if ((c & 0xF8) == 0xF0) char_len = 4;    // 4-byte
            else throw std::runtime_error("Invalid UTF-8 encoding");

            result.push_back(input.substr(i, char_len));
            i += char_len;
        }

        return result;
    }

    [[nodiscard]] int findTile(const Position pos) const {
        return pos.x + m_size * pos.y;
    }

    void generatePellet() {
        std::vector<size_t> poss_spawn_pos {};
        for (size_t i {0}; i < m_body.size(); ++i) {
            if (m_body[i].icon == " ")
                poss_spawn_pos.push_back(i);
        }
        const auto random_idx = Random::get<size_t>(0, (poss_spawn_pos.size()-1));
        const auto random_pos = poss_spawn_pos[random_idx];
        m_body[random_pos].icon = "▫";
    }

private:
    std::vector<Tile> m_body {};
    int m_size {};
};

void update(Player& player, const Action action, Board& board) {

    switch (action) {
        case Action::Quit:
            GAME = false;
        default:
            break;
    }

    player.setAction(action);
    player.update();
    board.update(player);
}

void render(const Board& board) {
    int length {};
    for (auto& [icon, pos]: board.getBody()) {
        if (icon == "┐") length = pos.x + 1;
    }

    int counter = {0};
    for (const auto& [icon, pos] : board.getBody()) {
        std::cout << icon;
        ++counter;
        if (counter % length == 0)
            std::cout << std::endl;
    }
}

void sleep(const int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


#pragma endregion


int main() {

    std::vector<std::string> b = {
        "┌────────────┐",
        "│            │",
        "│     ■      │",
        "│     ■  ▫   │",
        "│     ■      │",
        "│            │",
        "│            │",
        "│            │",
        "│            │",
        "│            │",
        "│            │",
        "│            │",
        "│            │",
        "└────────────┘"
    }; // YES this is quadratic 14 x 14
    auto player = Player{};
    Board board {player, 10};
    //
    setNonBlockingInput(true);
    while (GAME) {
        // Input
        auto input = inputPooling();

        // Logic
        update(player, input, board);

        // Render
        render(board);

        // Sleep
        sleep(TICK);
    }

    return 0;
}
