#include "FrameBuffer.h"

FrameBuffer::FrameBuffer(int size, std::string vp, std::string fp)
    : m_size(size), m_shader{vp, fp}{
    create();
    m_data.resize(m_size * m_size * 4);
}

FrameBuffer::~FrameBuffer() {
    destroy();
}

void FrameBuffer::draw() const {
    m_shader.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    m_shader.setValue("uTex", 0);

    m_ndc.draw();
}

void FrameBuffer::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, m_FBO);
}

void FrameBuffer::unbind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FrameBuffer::setData(const unsigned char* data, const int width, const int height) const {
    glBindTexture(GL_TEXTURE_2D, m_Tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
}

const std::vector<unsigned char> & FrameBuffer::getData() {
    glGetTextureImage(m_Tex, 0, GL_RGBA, GL_UNSIGNED_BYTE, m_data.size(), m_data.data());
    return m_data;
}

void FrameBuffer::create() {
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

void FrameBuffer::destroy() {
    if (m_FBO)
        glDeleteFramebuffers(1, &m_FBO);
    if (m_Tex)
        glDeleteTextures(1, &m_Tex);
    m_FBO = 0;
    m_Tex = 0;
}


