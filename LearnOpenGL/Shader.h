#pragma once

#include <string_view>
#include <filesystem>

namespace fs = std::filesystem;


class Shader {
public:
    Shader(const fs::path& vertexPath, const fs::path& fragmentPath);
    virtual ~Shader();

    void use() const;

    template<typename T>
    void setValue(std::string_view name, T val) const;

private:
    unsigned int ID;
};
