#include <print>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "Game.h"

#include <X11/Xlib.h>

#include "Assets.h"

// Draw a rectangle
float vertices[] = {
    // positions        // colors         // texture coords
    0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 2.0f, // top right
    0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, // bottom right
   -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom left
   -0.5f,  0.5f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 2.0f  // top left
};

unsigned int indices[] = {
    0, 1, 2, // first triangle
    2, 3, 0  // second triangle
};

float texCoords[] = {
    // x  y
    0.0f, 0.0f, // left bottom
    1.0f, 0.0f, // right bottom
    0.5f, 1.0f  // center top
};

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void createTexture(unsigned int& texture, const std::string_view path, const bool transparent = false) {
    // Texture stuff
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set the texture wrapping/filtering options (on currently bound texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // load and generate the texture
    int width, height, nrChannels;
    unsigned char *data = stbi_load(assets::path(path).c_str(), &width, &height,
    &nrChannels, 0);
    if (data)
    {
        if (transparent)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        else
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::println("Failed to load texture");
    }
    stbi_image_free(data);
}

Game::Game() {
    glfwInit();

    // Define OpenGL version (4.6)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create our first window
    m_window = glfwCreateWindow(800, 600, "LearnOP", nullptr, nullptr);
    if(m_window == nullptr) {
        std::println("Failed to create GLFW window!");
        return;
    }

    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    if(!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::println("Failed to initialize GLAD");
        return;
    }

    // Max Vertex Attributes
    int nrAttributes;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &nrAttributes);
    std::println("Maximum nr of vertex attributes supported : {}", nrAttributes);

    // Texture stuff
    stbi_set_flip_vertically_on_load(true);
    createTexture(texture1, m_texture_path1);
    createTexture(texture2, m_texture_path2, true);

    // Shader (important to do it here -> we have to have a OpenGL context active!)
    m_shader = std::make_unique<Shader>(m_vertex_shader_path, m_fragment_shader_path);

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);
    glGenBuffers(1, &m_EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(m_VAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), static_cast<void *>(nullptr));
    glEnableVertexAttribArray(0);

    // color attributes
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void *>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texture attributes
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), reinterpret_cast<void *>(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // uncomment this call to draw in wireframe polygons.
    // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture1);
    glActiveTexture(GL_TEXTURE0+1);
    glBindTexture(GL_TEXTURE_2D, texture2);
}

void Game::run() const {
    // render loop
    while(!glfwWindowShouldClose(m_window)) {
        // input
        processInput(m_window);

        // rendering commands
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Shader hot reload
        // TODO maybe add hot key for hot reload
        m_shader->checkReload();

        // draw our first triangle
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        // swap and check
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

Game::~Game() {
    glDeleteVertexArrays(1,&m_VAO);
    glDeleteBuffers(1, &m_VBO);
    glDeleteBuffers(1, &m_EBO);
    m_shader.reset();
    glfwTerminate();
}
