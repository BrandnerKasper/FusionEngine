#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GLFW/glfw3.h>

#define SENSITIVITY 0.1f
#define SPEED 5.5f
#define MAX_PITCH_UP 89.0f
#define MIN_PITCH_UP -89.0f
#define MIN_FOV 5.0f
#define MAX_FOV 120.0f

class Camera {
public:
    enum Move {
        Up,
        Down,
        Left,
        Right,
        MaxMoveOptions,
    };

    static void mouse_callback(GLFWwindow *window, double xpos, double ypos);
    static void scroll_callback(GLFWwindow* window, double xOffset, double yOffset);
    static void move(Move dir, double delta);

    static glm::mat4 getView() {return glm::lookAt(m_cameraPos, m_cameraPos + m_cameraFront, m_cameraUp);}
    static glm::mat4 getProjection(const float aspect) {return glm::perspective(glm::radians(static_cast<float>(m_fov)), aspect, 0.1f, 100.0f);}


private:
    static inline glm::vec3 m_cameraPos {glm::vec3(0.0f, 0.0f, 3.0f)};
    static inline glm::vec3 m_cameraFront {glm::vec3(0.0f, 0.0f, -1.0f)};;
    static inline glm::vec3 m_cameraUp {glm::vec3(0.0f, 1.0f, 0.0f)};;
    // rotate
    static inline bool m_firstMouse {true};
    static inline float m_lastX = 400, m_lastY = 300;
    static inline float m_yaw = -90.0f, m_pitch = 0.0f;
    // zoom
    static inline double m_fov {45.0f};
};
