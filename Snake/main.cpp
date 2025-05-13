#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

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

#define TICK 400
#define START_POSITION Position {6, 2}

struct Position {
    int x {};
    int y {};

    Position& operator+=(const Position& other) {
        x += other.x;
        y += other.y;
        return *this;
    }
};

using Tile = std::string;
using Board = std::vector<std::vector<Tile>>;

std::vector<std::string> splitUTF8Chars(const std::string& input) {
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


Board convert(std::vector<std::string> board) {
    Board b = {};
    for (auto& line : board) {
        std::vector<Tile> l = splitUTF8Chars(line);
        b.push_back(l);
    }
    return b;
}


struct Player {
    Tile icon = "■";
    Position pos = START_POSITION;

    void move(const Position& dir) {
        pos += dir;
    }
};


bool update(Player& player, const Action action, Board& board) {
    Position dir = {0, 0};

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
        case Action::Quit:
            return false;
        default:
            break;
    }

    // Move player
    board[player.pos.y][player.pos.x] = " ";
    player.move(dir);
    if (board[player.pos.y][player.pos.x] != " ") return false;
    board[player.pos.y][player.pos.x] = player.icon;

    return true;
}


void render(const Board& board) {
    for (auto& line : board) {
        for (const auto& icon : line) {
            std::cout << icon;
        }
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
        "│        ▫   │",
        "└────────────┘"
    };
    auto board = convert(b);

    auto player = Player{};

    setNonBlockingInput(true);
    bool game = true;
    while (game) {
        // Input
        auto input = inputPooling();

        // Logic
        game = update(player, input, board);

        // Render
        render(board);

        // Sleep
        sleep(TICK);
    }

    return 0;
}
