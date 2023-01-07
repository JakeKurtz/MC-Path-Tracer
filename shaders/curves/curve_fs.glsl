#version 330 core
out vec4 FragColor;

in vec3 FragPos;

uniform vec3 color;

void main()
{
    //if (mode == 1) FragColor = vec4(1.0, 0.64705, 0.21960, 1.0);
    FragColor = vec4(color, 1.0);
}

