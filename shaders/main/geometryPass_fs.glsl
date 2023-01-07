#version 330 core
layout(location = 0) out vec3 gPosition;
layout(location = 1) out vec3 gNormal;
layout(location = 2) out vec3 gAlbedo;
layout(location = 3) out vec3 gMetallicRoughAO;
layout(location = 4) out vec3 gEmissive;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;
in mat3 TBN;

uniform vec3 base_color_factor;
uniform vec3 emissive_color_factor;
uniform float roughness_factor;
uniform float metallic_factor;

uniform sampler2D base_color_texture;
uniform sampler2D normal_texture;
uniform sampler2D occlusion_texture;
uniform sampler2D emissive_texture;
uniform sampler2D metallic_roughness_texture;
uniform sampler2D roughness_texture;
uniform sampler2D metallic_texture;

uniform bool base_color_texture_sample;
uniform bool normal_texture_sample;
uniform bool occlusion_texture_sample;
uniform bool emissive_texture_sample;
uniform bool metallic_roughness_texture_sample;
uniform bool roughness_texture_sample;
uniform bool metallic_texture_sample;

void main()
{
    gPosition = FragPos;
    
    // BASE COLOR
    if (base_color_texture_sample)
        gAlbedo = texture(base_color_texture, TexCoords).rgb;
    else
        gAlbedo = base_color_factor;

    // NORMAL
    if (normal_texture_sample) {
        gNormal = texture(normal_texture, TexCoords).rgb;
        gNormal = gNormal * 2.0 - 1.0;
        gNormal = normalize(TBN * gNormal);
    }
    else
        gNormal = normalize(Normal);

    // EMISSIVE
    if (emissive_texture_sample)
        gEmissive = texture(emissive_texture, TexCoords).rgb;
    else
        gEmissive = vec3(0.f);

    if (metallic_roughness_texture_sample) {
        gMetallicRoughAO = texture(metallic_roughness_texture, TexCoords).rgb;
    } else {
        // OCCLUSION
        if (occlusion_texture_sample)
            gMetallicRoughAO.r = texture(occlusion_texture, TexCoords).r;
        else
            gMetallicRoughAO.r = 1.f;

        // ROUGHNESS
        if (roughness_texture_sample)
            gMetallicRoughAO.g = texture(roughness_texture, TexCoords).r;
        else
            gMetallicRoughAO.g = roughness_factor;

        // METALLIC
        if (metallic_texture_sample)
            gMetallicRoughAO.b = texture(metallic_texture, TexCoords).r;
        else
            gMetallicRoughAO.b = metallic_factor;
    }
}