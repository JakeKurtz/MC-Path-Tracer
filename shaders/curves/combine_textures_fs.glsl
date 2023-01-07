#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex_0;
uniform sampler2D tex_1;
//uniform sampler2D mask;

void main()
{
    vec3 col_0 = texture(tex_0, TexCoords).rgb;
    vec3 col_1 = texture(tex_1, TexCoords).rgb;
    float a = texture(tex_1, TexCoords).a;

    FragColor = vec4(a,0,0, 1.f);
    //FragColor = vec4(col_0*(1.f-a) + col_1*a, 1.f);
}

