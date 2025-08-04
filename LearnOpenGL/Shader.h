#pragma once

#include <glad/glad.h>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;


class Shader {
public:
    Shader(const fs::path& vertexPath, const fs::path& fragmentPath);

    void use() const;

    void setBool(std::string_view name, bool val) const;
    void setInt(std::string_view name, int val) const;
    void setFloat(std::string_view name, float val) const;

public:
    unsigned int ID;

private:
    static std::string loadShaderAsset(const fs::path& asset);
};
