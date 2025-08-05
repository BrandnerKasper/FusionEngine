#pragma once

#include <string>
#include <filesystem>

namespace fs = std::filesystem;


class Shader {
public:
    Shader(const fs::path& vertexPath, const fs::path& fragmentPath);
    virtual ~Shader();

    void use() const;

    template<typename T>
    void setValue(const std::string& name, T val) const;

private:
    unsigned int ID;
};

template<typename T>
void Shader::setValue(const std::string& name, const T val) const {
    GLint loc = glGetUniformLocation(ID, name.c_str());
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int>) {
        glUniform1i(loc, static_cast<int>(val));
    } else if constexpr (std::is_same_v<T, float>) {
        glUniform1f(loc, val);
    } else {
        static_assert(sizeof(T) == 0, "Unsupported uniform type for setVal");
    }
}