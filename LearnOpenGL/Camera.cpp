#include <print>
#include "Camera.h"


Camera::Camera(const glm::vec3 start_pos, const bool is_flying)
    : m_pos{start_pos}, m_isFlying(is_flying){}

void Camera::processMouseMovement(const double xpos, const double ypos) {
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
    m_front = glm::normalize(direction);
}

void Camera::processMouseWheel(double xOffset, double yOffset) {
    m_fov -= yOffset;
    if (m_fov < MIN_FOV)
        m_fov = MIN_FOV;
    if (m_fov > MAX_FOV)
        m_fov = MAX_FOV;
}

void Camera::move(const Move dir, const double delta) {
    const float cameraSpeed = SPEED * delta;
    switch (dir) {
        case Forward:
            if (m_isFlying)
                m_pos += cameraSpeed * m_front;
            else
                m_pos += cameraSpeed * glm::vec3(m_front.x, 0, m_front.z);
            break;
        case Backward:
            if (m_isFlying)
                m_pos -= cameraSpeed * m_front;
            else
                m_pos -= cameraSpeed * glm::vec3(m_front.x, 0, m_front.z);
            break;
        case Left:
            m_pos -= glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
            break;
        case Right:
            m_pos += glm::normalize(glm::cross(m_front, m_up)) * cameraSpeed;
            break;
        default:
            std::println("Camera movement option {} is not valid!", static_cast<int>(dir));
    }
}
