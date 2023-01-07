#version 330 core

out vec4 frag_out;

in vec3 near_point; // near_point calculated in vertex shader
in vec3 far_point; // far_point calculated in vertex shader
in vec2 tex_coords;

uniform vec2 tex_dim;
uniform vec2 scr_dim;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform float grid_scale;
uniform float max_dist_line_width;
uniform float min_dist_line_width;

uniform float near;
uniform float far;

uniform int fade_mode;
uniform bool highlight_axis;

uniform vec4 color;

uniform sampler2D depth_tex;
uniform sampler2D wireframe_tex;

const float EPSILON_ALPHA = 0.001;

float remap(float high1, float low1, float high2, float low2, float value) {
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

vec2 remap(vec2 high1, vec2 low1, vec2 high2, vec2 low2, vec2 value) {
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1);
}

float compute_depth(vec3 pos) {
    vec4 clip_space_pos = projection * view * model * vec4(pos.xyz, 1.0);
    return clip_space_pos.z / (2.f*clip_space_pos.w) + 0.5; // [0, 1]
}

float compute_linear_depth(vec3 pos) {
    vec4 clip_space_pos = projection * view * model * vec4(pos.xyz, 1.0);
    float clip_space_depth = (clip_space_pos.z / clip_space_pos.w); // [-1, 1]
    float linear_depth = (2.0 * near * far) / (far + near - clip_space_depth * (far - near)); // get linear value between 0.01 and 100
    return linear_depth / far; // normalize
}

vec4 grid(vec3 pos, float scale, float line_width) {

    vec2 coord = pos.xz * scale; // use the scale variable to set the distance between the lines
    vec2 derivative = fwidth(coord);
    vec2 derivative2 = fwidth(pos.xz);

    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;

    float _line = min(grid.x, grid.y);

    float delta = fwidth(_line);
    float alpha = 1.0 - smoothstep(line_width - delta, line_width, _line);

    vec3 grid_color = color.rgb;

    float axis_scale = 3;
    float axis_width = line_width * axis_scale;

    float minimumz = min(derivative2.y, 1) * axis_width * 0.5;
    float minimumx = min(derivative2.x, 1) * axis_width * 0.5;

    if (highlight_axis) {
        // z axis
        if (pos.x > -1 * minimumx && pos.x < 1 * minimumx) {
            grid_color = vec3(0.0, 0.0, 1.0);
            alpha = 1.0 - smoothstep(axis_width - delta, axis_width, _line);
        }
        // x axis
        if (pos.z > -1 * minimumz && pos.z < 1 * minimumz) {
            grid_color = vec3(1.0, 0.0, 0.0);
            alpha = 1.0 - smoothstep(axis_width - delta, axis_width, _line);
        }
    }

    vec4 color = vec4(grid_color, alpha*color.a);

    return color;
}

void main() {
    vec2 tex_coords_remap = remap(tex_coords, vec2(0.f), vec2(1.f), vec2(0.f), vec2(tex_dim / scr_dim));

    float t = -near_point.y / (far_point.y - near_point.y);
    vec3 grid_pos = near_point + t * (far_point - near_point);
    
    float fading = 1.f;
    
    if (fade_mode == 0) {
        vec3 clip_space_pos = vec3(vec4(0.f, 0.f, 0.f, 1.f) * model);
        float dist_from_origin = distance(clip_space_pos, grid_pos);
        fading = remap(far, near, 0, 1, dist_from_origin);
    }
    else if (fade_mode == 1) {
        float linear_depth = compute_linear_depth(grid_pos);
        fading = max(0, (0.65 - linear_depth));
    }

    float line_width = clamp(max_dist_line_width * fading, min_dist_line_width, max_dist_line_width);

    vec4 grid_value = grid(grid_pos, grid_scale, line_width);

    float depth_0 = texture(depth_tex, tex_coords_remap).r;
    float depth_1 = compute_depth(grid_pos);

    vec4 wireframe = texture(wireframe_tex, tex_coords_remap);

    float grid_alpha = grid_value.a * (t > 0.f ? 1.f : 0.f) * fading;
    
    if (depth_0 > depth_1 && grid_alpha > EPSILON_ALPHA) {
        vec3 color = mix(wireframe.rgb, grid_value.rgb, grid_alpha);
        float alpha = mix(wireframe.a, 1.f, grid_alpha);

        frag_out = vec4(color, alpha);
    }
    else {
        frag_out = wireframe;
    }

    //frag_out = vec4(grid_pos, 1.f);
}


