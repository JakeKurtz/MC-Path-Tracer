#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 near_point;
out vec3 far_point;

out vec2 tex_coords;

void main()
{
    mat4 inv_mat = inverse(model) * inverse(view) * inverse(projection);

    vec4 unproject_near = inv_mat * vec4(aPos.xy, 0.0, 1.0);
    vec4 unproject_far = inv_mat * vec4(aPos.xy, 1.0, 1.0);

    near_point = unproject_near.xyz / unproject_near.w;
    far_point = unproject_far.xyz / unproject_far.w;

    gl_Position = vec4(aPos, 1.0);

    tex_coords = aTexCoords;
}