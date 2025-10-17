#include <iostream>
#include <format>

#include "Texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../extern/stb_image.h"
#include "../assets.h"

Texture::Texture(const std::string_view path) {
    init(path);
}

void Texture::bind() const {
    glBindTexture(GL_TEXTURE_2D, m_ID);
}

void Texture::init(const std::string_view path) {
    glGenTextures(1, &m_ID);
    glBindTexture(GL_TEXTURE_2D, m_ID);

    // set the texture wrapping/filtering options (on currently bound texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load(assets::path(path).c_str(), &width, &height,
    &nrChannels, 0);

    if (data) {
        // Only have RGBA files via .png
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D); // not sure if needed
    }
    else
        throw std::runtime_error(std::format("Failed to load texture at path {}.", path));
    stbi_image_free(data);
    // unbind texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

