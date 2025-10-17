#pragma once
#include <string>

class IWindow {
public:
    virtual ~IWindow() = default;
    [[nodiscard]] virtual void* get() const = 0;
    [[nodiscard]] virtual bool shouldClose() const = 0;
    virtual void setTitle(const std::string& t) const = 0;

    // convenience, not virtual
    template <class T>
    T* handleAs() const {
        return static_cast<T*>(get());
    }
};