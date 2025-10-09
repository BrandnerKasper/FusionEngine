#pragma once
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <unordered_map>

#include "IInput.h"
#include "IInput.h"


class TerminalInput final : public IInput{
public:
    TerminalInput();
    ~TerminalInput() override;

    void update() override;

private:
    termios m_old_{}, m_new_{};
    std::unordered_map<Action, int> m_bind{
        {Quit, 'q'},
        {Up, 'w'},
        {Left, 'a'},
        {Down, 's'},
        {Right, 'd'},
        {Pause, 'p'},
    };
};
