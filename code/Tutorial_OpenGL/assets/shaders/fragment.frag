#version 460 core
out vec4 fragColor;

in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float ourMix;

void main() {
    fragColor = mix(texture(texture1, TexCoord), texture(texture2, vec2(TexCoord.x*1.0, TexCoord.y)), ourMix);
}