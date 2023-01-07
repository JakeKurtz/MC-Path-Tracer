#version 330 core

out vec4 frag_out;

in vec3 FragPos;

uniform vec4 color;

void main()
{
    frag_out = color;
}

