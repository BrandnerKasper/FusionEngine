#include "input.h"
#include <iostream>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

void Input::setNonBlockingInput(bool enable) {
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

void Input::inputPooling(Action &input) {
    if (const char ch = static_cast<char>(getchar()); ch != EOF) {
        switch (ch) {
            case 'w': case 'W':
                std::cout << "UP\n";
                input = Action::Up;
                break;
            case 'a': case 'A':
                std::cout << "LEFT\n";
                input = Action::Left;
                break;
            case 's': case 'S':
                std::cout << "DOWN\n";
                input = Action::Down;
                break;
            case 'd': case 'D':
                std::cout << "RIGHT\n";
                input = Action::Right;
                break;
            case 'q': case 'Q':
                std::cout << "QUIT\n";
                input =  Action::Quit;
                break;
            default:
                std::cout << "DO THE LAST INPUT ACTION!\n";
        }
    }
}

Types::Position Input::getDirectionFromAction(const Input::Action action) {
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


