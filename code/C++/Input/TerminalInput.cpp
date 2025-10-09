#include "TerminalInput.h"

#include <cstdio>


TerminalInput::TerminalInput() {
    tcgetattr(STDIN_FILENO, &m_old_);
    m_new_ = m_old_;
    m_new_.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &m_new_);
    fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
}

TerminalInput::~TerminalInput() {
    tcsetattr(STDIN_FILENO, TCSANOW, &m_old_);
}

void TerminalInput::update() {
    int c {getchar()};
    while (c!=EOF) {
        for (auto [act, key]: m_bind) {
            if (c == key) m_curr = act;
        }
        c = getchar();
    }
}
