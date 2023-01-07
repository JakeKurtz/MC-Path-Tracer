#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D image;
  
uniform bool horizontal;
uniform float r; // blur radius

void main()
{
    float x, y, rr = r * r, d, w, w0;
    vec2 p = TexCoords;
    vec4 col = vec4(0.0, 0.0, 0.0, 0.0);
    w0 = 0.5135 / pow(r, 0.96);

    vec2 tex_size = textureSize(image, 0).xy;
    float xs = tex_size.x;
    float ys = tex_size.y;

    if (horizontal) {
        for (d = 1.0 / xs, x = -r, p.x += x * d; x <= r; x++, p.x += d)
        {
            w = w0 * exp((-x * x) / (2.0 * rr));
            col += texture(image, p) * w;
        }
    }
    else {
        for (d = 1.0 / ys, y = -r, p.y += y * d; y <= r; y++, p.y += d)
        {
            w = w0 * exp((-y * y) / (2.0 * rr));
            col += texture(image, p) * w;
        }
    }
    FragColor = col;
}