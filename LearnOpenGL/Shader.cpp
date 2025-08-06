#include <fstream>
#include <sstream>
#include <format>
#include <print>
#include <glad/glad.h>

#include "Shader.h"
#include "Assets.h"

std::string loadShaderAsset(const fs::path& asset) {
    std::ifstream file(assets::path(asset), std::ios::in);
    if (!file.is_open())
        throw std::runtime_error(std::format("Failed to open shader file {}.", asset.c_str()));

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

unsigned int createShader(const fs::path& asset, GLuint type) {
    const auto source {loadShaderAsset(asset)};
    const auto s {source.c_str()};
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &s, nullptr);
    glCompileShader(shader);
    // Check
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::println("ERROR::SHADER::{}::COMPILATION_FAILED {}", type == GL_VERTEX_SHADER ? "Vertex" : "Fragment", infoLog);
    }
    return shader;
}


Shader::Shader(const fs::path &vertexPath, const fs::path &fragmentPath) {
    const auto vertexShader = createShader(vertexPath, GL_VERTEX_SHADER);
    const auto fragmentShader = createShader(fragmentPath, GL_FRAGMENT_SHADER);

    // Shader Program
    ID =  glCreateProgram();
    glAttachShader(ID, vertexShader);
    glAttachShader(ID, fragmentShader);
    glLinkProgram(ID);
    // Check
    int success;
    char infoLog[512];
    glGetProgramiv(ID, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(ID, 512, nullptr, infoLog);
        std::println("ERROR::SHADER::PROGRAM::COMPILATION_FAILED {}", infoLog);
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

Shader::~Shader() {
    glDeleteProgram(ID); // TODO: right now that get's called after gltf terminate, should be before!! -> make class for window or game
}

void Shader::use() const {
    glUseProgram(ID);
}