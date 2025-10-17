#pragma once

#include <string>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace fs = std::filesystem;


class Shader {
public:
    Shader(fs::path  vertexPath, fs::path  fragmentPath);
    virtual ~Shader();

    void use() const;

    template<typename T>
    void setValue(const std::string& name, T val) const;

    void checkReload();

private:
    void create();
    void reload();

private:
    unsigned int ID;
    fs::path m_vertexShader;
    fs::path m_fragmentShader;
    fs::file_time_type m_vsTime, m_fsTime;
};

template<typename T>
void Shader::setValue(const std::string& name, const T val) const {
    GLint loc = glGetUniformLocation(ID, name.c_str());
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, int>) {
        glUniform1i(loc, static_cast<int>(val));
    } else if constexpr (std::is_same_v<T, float>) {
        glUniform1f(loc, val);
    } else if constexpr (std::is_same_v<T, glm::mat4>) {
        glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(val));
    } else {
        static_assert(sizeof(T) == 0, "Unsupported uniform type for setVal");
    }
}