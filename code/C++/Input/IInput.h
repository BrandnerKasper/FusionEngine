#pragma once

struct IInput {
    enum Action {Quit, Up, Down, Left, Right, Pause};

    virtual ~IInput() = default;

    virtual void update() = 0;
    [[nodiscard]] virtual bool pressed(const Action a) const {return m_curr == a;}
    virtual Action getAction() const {return m_curr;}
    virtual void clear() {m_curr = Up;}

protected:
    Action m_curr {Up};
};