#include "Game.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <print>

void tryGLM() {
    glm::vec4 vec {1.0f, 1.0f, 1.0f, 1.0f};
    auto trans {glm::mat4{1.0f}};
    trans = glm::translate(trans, glm::vec3{1.0f, 1.0f, 0.0f});
    auto res = trans * vec;
    std::println("Our res X{} Y{} Z{}.", res.x, res.y, res.z);
}


int main() {
    const auto game {std::make_unique<Game>()};
    game->run();
    // tryGLM();
}

