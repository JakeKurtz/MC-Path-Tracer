#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform float cam_exposure;

uniform samplerCube skybox;

vec3 reinhard(vec3 v)
{
    return v / (1.0f + v);
}

void main()
{    
    vec3 color = texture(skybox, TexCoords).rgb * cam_exposure;

    //const float gamma = 2.2;
    //color = pow(color, vec3(1.0 / gamma));
    
    FragColor = vec4(reinhard(color), 1.0);
}