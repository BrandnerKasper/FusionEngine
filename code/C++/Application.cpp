#include <iostream>

#include "Application.h"


Application::Application() {
    init();
    Input::m_window = m_window;
    // m_renderer = std::make_unique<Renderer>(m_window);
    m_ascii_renderer = std::make_unique<ASCIIRenderer>();
    m_neural_renderer = std::make_unique<NeuralRenderer>(m_window);
}

Application::~Application() {
    Input::m_window = nullptr;
}

// OpenGL call backs
void framebuffer_size_callback(GLFWwindow* window, const int width, const int height) {
    glViewport(0, 0, width, height);
}

void Application::init() {
    glfwInit();

    // Define OpenGL version (4.6)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create our first window
    m_window = glfwCreateWindow(m_width, m_height, m_title.c_str(), nullptr, nullptr);
    if(m_window == nullptr)
        throw std::runtime_error("Failed to create GLFW window!");

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        throw std::runtime_error("Failed to initialize GLAD");
    // Depth testing
    glEnable(GL_DEPTH_TEST);
}

void Application::run() {
    while(!glfwWindowShouldClose(m_window)) {
        // Delta Time
        const auto currentTime {glfwGetTime()};
        m_deltaTime = currentTime - m_last_frame;
        m_last_frame = currentTime;

        processInput();
        update();
        render();
        genData();
    }
}

void Application::processInput() {
    if (Input::pressed(Input::Quit))
        glfwSetWindowShouldClose(m_window, true);
    if (Input::pressed(Input::Pause))
        m_current_action = Input::Pause;
    // TODO EVENT SYSTEM to subscribe onto Input event
    if (Input::pressed(Input::Up))
        m_current_action = Input::Up;

    if (Input::pressed(Input::Down))
        m_current_action = Input::Down;

    if (Input::pressed(Input::Left))
        m_current_action = Input::Left;

    if (Input::pressed(Input::Right))
        m_current_action = Input::Right;
}

void Application::update() {
    if (m_current_action == Input::Pause)
        return;
    if (auto play = m_game.run(m_deltaTime, m_current_action))
        board_state = m_game.getBoardState();
    else {
        m_current_action = Input::Up;
        m_game.reset();
    }
}

void Application::render() {
    // Terminal render
    m_last_render += m_deltaTime;
    if (m_last_render >= Settings::Render::frame_time) {
        terminalRender(board_state);
        m_last_render -= Settings::Render::frame_time;
    }
    // openGLRender(board_state);
    neuralRender(board_state);
}


void Application::terminalRender(const std::string_view board) const {
    m_ascii_renderer->draw(board);
}

// void Application::openGLRender(const std::string_view board) const {
//     m_renderer->draw(board);
// }

void Application::neuralRender(const std::string_view board) const {
    m_neural_renderer->draw(board);
}

// TODO: Only generate data if we use the normal renderer
void Application::genData() {
    if (!generate)
        return;
    static int count {};
    if (count == Settings::Data::amount)
        return;
    if (prev_board_state != board_state) {
        prev_board_state = board_state;
        m_ascii_renderer->generateData("in", count);
        // m_renderer->generateData("out", count);
        ++count;
    }
}
