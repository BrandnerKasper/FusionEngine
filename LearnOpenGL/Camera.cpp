#include <print>
#include "Camera.h"


void Camera::mouse_callback(GLFWwindow *window, const double xpos, const double ypos) {
    if (m_firstMouse) {
        m_lastX = xpos;
        m_lastY = ypos;
        m_firstMouse = false;
    }

    float xOffset = xpos - m_lastX;
    float yOffset = m_lastY - ypos;
    m_lastX = xpos;
    m_lastY = ypos;

    xOffset *= SENSITIVITY;
    yOffset *= SENSITIVITY;

    m_yaw += xOffset;
    m_pitch += yOffset;

    if (m_pitch > MAX_PITCH_UP)
        m_pitch = MAX_PITCH_UP;
    if (m_pitch < MIN_PITCH_UP)
        m_pitch = MIN_PITCH_UP;

    glm::vec3 direction;
    direction.x = std::cos(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    direction.y = std::sin(glm::radians(m_pitch));
    direction.z = std::sin(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    m_cameraFront = glm::normalize(direction);
}

void Camera::scroll_callback(GLFWwindow *window, double xOffset, double yOffset) {
    m_fov -= yOffset;
    if (m_fov < MIN_FOV)
        m_fov = MIN_FOV;
    if (m_fov > MAX_FOV)
        m_fov = MAX_FOV;
}

void Camera::move(const Move dir, const double delta) {
    const float cameraSpeed = SPEED * delta;
    switch (dir) {
        case Up:
            m_cameraPos += cameraSpeed * m_cameraFront;
            break;
        case Down:
            m_cameraPos -= cameraSpeed * m_cameraFront;
            break;
        case Left:
            m_cameraPos -= glm::normalize(glm::cross(m_cameraFront, m_cameraUp)) * cameraSpeed;
            break;
        case Right:
            m_cameraPos += glm::normalize(glm::cross(m_cameraFront, m_cameraUp)) * cameraSpeed;
            break;
        default:
            std::println("Camera movement option {} is not valid!", static_cast<int>(dir));
    }
}
