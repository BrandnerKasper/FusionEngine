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

glm::mat4 Camera::getView2() {
    // 1. Position = known
    glm::vec3 position = {m_pos};
    glm::vec3 target {0, 0, 0};
    glm::vec3 worldUp {0, 1, 0};
    // 2. Calculate cameraDirection
    glm::vec3 zaxis = glm::normalize(position - target);
    // 3. Get positive right axis vector
    glm::vec3 xaxis = glm::normalize(glm::cross(glm::normalize(worldUp), zaxis));
    // 4. Calculate camera up vector
    glm::vec3 yaxis = glm::cross(zaxis, xaxis);

    // Create translation and rotation matrix
    // In glm we access elements as mat[col][row] due to column-major layout
    glm::mat4 translation = glm::mat4(1.0f); // Identity matrix by default
    translation[3][0] = -position.x; // Fourth column, first row
    translation[3][1] = -position.y;
    translation[3][2] = -position.z;
    glm::mat4 rotation = glm::mat4(1.0f);
    rotation[0][0] = xaxis.x; // First column, first row
    rotation[1][0] = xaxis.y;
    rotation[2][0] = xaxis.z;
    rotation[0][1] = yaxis.x; // First column, second row
    rotation[1][1] = yaxis.y;
    rotation[2][1] = yaxis.z;
    rotation[0][2] = zaxis.x; // First column, third row
    rotation[1][2] = zaxis.y;
    rotation[2][2] = zaxis.z;

    // Return lookAt matrix as combination of translation and rotation matrix
    return rotation * translation; // Remember to read from right to left (first translation then rotation)
}
