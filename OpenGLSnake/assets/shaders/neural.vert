#version 460 core

layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;

void main(){
//    vUV = vec2(aUV.x*-1, aUV.y);
    vUV = aUV;
    gl_Position = vec4(aPos,0,1);
}
