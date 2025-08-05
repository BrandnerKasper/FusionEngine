#version 460 core
out vec4 fragColor;

in vec3 ourColor;
in vec4 ourPos;

void main() {
//    fragColor = vec4(ourColor, 1.0);
    fragColor = ourPos; // negative vals get clamped to 0.0 = black!
}