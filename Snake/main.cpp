#include <iostream>
#include <vector>
#include <string>


int main() {


    std::vector<std::string> board = {
        "┌────────────┐",
        "│     ■      │",
        "│            │",
        "│        ▫   │",
        "└────────────┘"
    };

    for (auto& line : board) {
        std::cout << line << std::endl;
    }

}
