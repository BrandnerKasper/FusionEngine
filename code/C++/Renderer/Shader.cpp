#include <iostream>
#include <format>
#include <fstream>
#include <sstream>

#include "Shader.h"
#include "../assets.h"

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
        throw std::runtime_error(std::format("ERROR::SHADER::{}::COMPILATION_FAILED {}", type == GL_VERTEX_SHADER ? "Vertex" : "Fragment", infoLog));
    }
    return shader;
}

Shader::Shader(fs::path vertexPath, fs::path fragmentPath)
    : m_vertexShader{std::move(vertexPath)}, m_fragmentShader{std::move(fragmentPath)} {
    create();
}

Shader::~Shader() {
    glDeleteProgram(m_ID);
}

void Shader::use() const {
    glUseProgram(m_ID);
}

void Shader::create() {
    const auto vertexShader = createShader(m_vertexShader, GL_VERTEX_SHADER);
    const auto fragmentShader = createShader(m_fragmentShader, GL_FRAGMENT_SHADER);

    // Shader Program
    m_ID = glCreateProgram();
    glAttachShader(m_ID, vertexShader);
    glAttachShader(m_ID, fragmentShader);
    glLinkProgram(m_ID);
    // Check
    int success;
    char infoLog[512];
    glGetProgramiv(m_ID, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(m_ID, 512, nullptr, infoLog);
        throw std::runtime_error(std::format("ERROR::SHADER::PROGRAM::COMPILATION_FAILED {}", infoLog));
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    // don't know if we should set that here
    use();
}
