#pragma once

#include "types.h"

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

    Types::Position getDirectionFromAction(Input::Action action);
}



