#version 330 core
out vec4 frag_out;
in vec2 uv;

uniform vec2 tex_dim;
uniform vec2 scr_dim;

uniform sampler2D tex;

vec2 remap(vec2 value, vec2 min1, vec2 max1, vec2 min2, vec2 max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

void main()
{
    vec2 tex_coords_remap = remap(uv, vec2(0.f), vec2(1.f), vec2(0.f), vec2(tex_dim / scr_dim));
    frag_out = texture(tex, tex_coords_remap);
}

