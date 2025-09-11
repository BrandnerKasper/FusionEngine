#include <glad/glad.h>

#include "RenderTexture.h"
#include "../assets.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../extern/stb_image_write.h"

RenderTexture::RenderTexture() {
    create();
    pixels.resize(32*32*4);
}

void RenderTexture::draw() {
    m_present.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    m_present.setValue("uTex", 0);

    m_ndc.draw();
}

void saveTexAsImg(const std::string& filename, const int w, const int h, const std::vector<unsigned char>& rgba) {
    // Flip vertically for correct orientation
    std::vector<unsigned char> flipped(w * h * 4);
    for (int y = 0; y < h; ++y) {
        memcpy(&flipped[y * w * 4],
               &rgba[(h - 1 - y) * w * 4],
               w * 4);
    }

    // stb_image_write wants 3 channels for jpg
    // std::vector<unsigned char> rgb(w * h * 3);
    // for (int i = 0, j = 0; i < w * h; ++i) {
    //     rgb[j++] = flipped[i * 4 + 0];
    //     rgb[j++] = flipped[i * 4 + 1];
    //     rgb[j++] = flipped[i * 4 + 2];
    // }

    stbi_write_png(filename.c_str(), w, h, 4, rgba.data(), w*4);
}

void RenderTexture::getTextureImage() {
    glGetTextureImage(m_colorTex, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.size(), pixels.data());
    saveTexAsImg(data::path("test.png"), 32, 32, pixels);
}

void RenderTexture::create() {
    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_offW, m_offH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}
