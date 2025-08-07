#include <fstream>
#include <sstream>
#include <format>
#include <print>
#include <utility>
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


Shader::Shader(fs::path vertexPath, fs::path fragmentPath)
    : m_vertexShader{std::move(vertexPath)}, m_fragmentShader{std::move(fragmentPath)},
      m_vsTime{fs::last_write_time(assets::path(m_vertexShader))}, m_fsTime{fs::last_write_time(assets::path(m_fragmentShader))} {
    create();
}

Shader::~Shader() {
    glDeleteProgram(ID);
}

void Shader::use() const {
    glUseProgram(ID);
}

void Shader::checkReload() {
    const auto currVsTime {fs::last_write_time(assets::path(m_vertexShader))};
    const auto currFSTime {fs::last_write_time(assets::path(m_fragmentShader))};
    if (currVsTime != m_vsTime || currFSTime != m_fsTime) {
        std::println("Shader hot-reloaded!");
        m_vsTime = currVsTime;
        m_fsTime = currFSTime;
        reload();
    }
}

void Shader::create() {
    const auto vertexShader = createShader(m_vertexShader, GL_VERTEX_SHADER);
    const auto fragmentShader = createShader(m_fragmentShader, GL_FRAGMENT_SHADER);

    // Shader Program
    ID = glCreateProgram();
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
    // don't know if we should set that here
    use();
    setValue("texture1", 0);
    setValue("texture2", 1);
    setValue("ourMix", 0.2f);
}

void Shader::reload() {
    glDeleteProgram(ID);
    create();
}
