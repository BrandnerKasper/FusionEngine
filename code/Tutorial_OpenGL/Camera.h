#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#define SENSITIVITY 0.1f
#define SPEED 5.5f
#define MAX_PITCH_UP 89.0f
#define MIN_PITCH_UP -89.0f
#define MIN_FOV 5.0f
#define MAX_FOV 120.0f

class Camera {
public:
    enum Move {
        Forward,
        Backward,
        Left,
        Right,
        MaxMoveOptions,
    };

    explicit Camera(glm::vec3 start_pos = glm::vec3(0.0f, 0.0f, 3.0f), bool is_flying = true);

    void processMouseMovement(double xpos, double ypos);
    void processMouseWheel(double xOffset, double yOffset);
    void move(Move dir, double delta);

    [[nodiscard]] glm::mat4 getView() const {return glm::lookAt(m_pos, m_pos + m_front, m_up);}
    glm::mat4 getView2();
    [[nodiscard]] glm::mat4 getProjection(const float aspect) const {return glm::perspective(glm::radians(static_cast<float>(m_fov)), aspect, 0.1f, 100.0f);}


private:
    // flying or walking camer
    glm::vec3 m_pos {};
    bool m_isFlying {};
    glm::vec3 m_front {glm::vec3(0.0f, 0.0f, -1.0f)};;
    glm::vec3 m_up {glm::vec3(0.0f, 1.0f, 0.0f)};;
    // rotate
    bool m_firstMouse {true};
    float m_lastX = 400, m_lastY = 300;
    float m_yaw = -90.0f, m_pitch = 0.0f;
    // zoom
    double m_fov {45.0f};
};
