#include <fstream>
#include <sstream>
#include <format>
#include <print>

#include "Shader.h"
#include "Assets.h"


Shader::Shader(const fs::path &vertexPath, const fs::path &fragmentPath) {
    unsigned int v, f;

    // Vertex shader
    const auto vs {loadShaderAsset(vertexPath)};
    const auto vsSource {vs.c_str()};
    v = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(v, 1, &vsSource, nullptr);
    glCompileShader(v);
    // Check
    int success;
    char infoLog[512];
    glGetShaderiv(v, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(v, 512, nullptr, infoLog);
        std::println("ERROR::SHADER::VERTEX::COMPILATION_FAILED {}", infoLog);
    }

    // Fragment shader
    const auto fs {loadShaderAsset(fragmentPath)};
    const auto fsSource {fs.c_str()};
    f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(f, 1, &fsSource, nullptr);
    glCompileShader(f);
    // Check
    glGetShaderiv(f, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(f, 512, nullptr, infoLog);
        std::println("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED {}", infoLog);
    }

    // Shader Program
    ID =  glCreateProgram();
    glAttachShader(ID, v);
    glAttachShader(ID, f);
    glLinkProgram(ID);
    // Check
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(ID, 512, nullptr, infoLog);
        std::println("ERROR::SHADER::PROGRAM::COMPILATION_FAILED {}", infoLog);
    }
    // glUseProgram(shaderProgram);
    glDeleteShader(v);
    glDeleteShader(f);
}

void Shader::use() const {
    glUseProgram(ID);
}


std::string Shader::loadShaderAsset(const fs::path& asset) {
    std::ifstream file(assets::path(asset), std::ios::in);
    if (!file.is_open())
        throw std::runtime_error(std::format("Failed to open shader file {}.", asset.c_str()));

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}
