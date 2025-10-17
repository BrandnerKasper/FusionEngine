#pragma once

#include <string_view>
#include <glad/glad.h>

class Texture {
public:
    Texture(std::string_view path);

    void bind() const;

private:
    void init(std::string_view path);
private:
    unsigned int m_ID;
};

