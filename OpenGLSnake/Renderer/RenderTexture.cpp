#include <glad/glad.h>

#include "RenderTexture.h"
#include "../settings.h"


RenderTexture::RenderTexture(const int size)
    : m_size{size} {
    create();
    pixels.resize(m_size * m_size * 4);
}

RenderTexture::~RenderTexture() {
    destroy();
}

void RenderTexture::begin() {
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glViewport(0, 0, m_size, m_size);
}

void RenderTexture::draw() {
    m_present.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    m_present.setValue("uTex", 0);

    m_ndc.draw();
}

void RenderTexture::end(const int width, const int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);
}

const std::vector<unsigned char>& RenderTexture::getTextureImage() {
    glGetTextureImage(m_Tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.size(), pixels.data());
    return pixels;
}

void RenderTexture::create() {
    // Texture
    glGenTextures(1, &m_Tex);
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_size, m_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Important: set sampling so upscaling looks crisp (or change to LINEAR if you want)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // FBO
    glGenFramebuffers(1, &m_FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_Tex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        throw std::runtime_error("RenderTexture: framebuffer is incomplete");
    }

    // Unbind
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderTexture::destroy() {
    if (m_FBO)
        glDeleteFramebuffers(1, &m_FBO);
    if (m_Tex)
        glDeleteTextures(1, &m_Tex);
    m_FBO = 0;
    m_Tex = 0;
}
