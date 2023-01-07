#version 330 core

out vec4 FragColor;

in vec3 FragPos;

uniform bool enable_xray;

void main()
{
    FragColor = vec4(0.f);
}

