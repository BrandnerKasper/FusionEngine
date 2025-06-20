#pragma once

namespace Input {
    enum Action {
        Up,
        Left,
        Down,
        Right,
        Quit,
    };

    void setNonBlockingInput(bool enable);

    void inputPooling(Action& input);
}



