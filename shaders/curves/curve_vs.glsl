#version 330 core
layout(location = 0) in vec3 aPos;

out vec3 FragPos;
//out vec2 TexCoords;
//out vec3 Normal;
//out mat3 TBN;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    //vec3 T = normalize(vec3(model * vec4(aTangent, 0.0)));
    //vec3 B = normalize(vec3(model * vec4(aBitangent, 0.0)));
    //vec3 N = normalize(vec3(model * vec4(aNormal, 0.0)));
    //TBN = mat3(T, B, N);

    vec4 worldPos = model * vec4(aPos, 1.0);
    FragPos = vec3(worldPos);
    //TexCoords = aTexCoord;

    //mat3 normalMatrix = transpose(inverse(mat3(model)));
    //Normal = normalMatrix * aNormal;

    gl_Position = projection * view * worldPos;
}