#include <glad/glad.h>

#include "RenderTexture.h"

RenderTexture::RenderTexture() {
    create();
}

void RenderTexture::draw() {
    m_present.use();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    m_present.setValue("uTex", 0);

    m_ndc.draw();
}

void RenderTexture::create() {
    glGenTextures(1, &m_colorTex);
    glBindTexture(GL_TEXTURE_2D, m_colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_offW, m_offH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
}
