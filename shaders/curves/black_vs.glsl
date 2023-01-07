#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);
    gl_Position = projection * view * worldPos;
}