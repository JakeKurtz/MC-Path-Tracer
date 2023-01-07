#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform float cam_exposure;
uniform vec3 camPos;
uniform float lightSize;
uniform float searchAreaSize;
uniform int kernel_size;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D gMetallicRoughAO;
uniform sampler2D gEmissive;
uniform sampler2D gShadows;

uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;
uniform sampler2DArray shadowMaps;

#define M_PI 3.14159265358979323846264338327

const float MAX_REFLECTION_LOD = 4.0;
const int NR_DIR_LIGHTS = 32;
const int NR_POINT_LIGHTS = 32;
const float g_MinVariance = 0.0001;

layout(std140) uniform ExampleBlock
{
    mat4 lightSpaceMatrix[NR_DIR_LIGHTS];
};

struct PointLight {
    vec3 position;
    vec3 color;
    float intensity;
};

struct DirectionalLight {
    vec3 direction;
    vec3 color;
    float intensity;
};

uniform PointLight pnt_lights[NR_POINT_LIGHTS];
uniform DirectionalLight dir_lights[NR_DIR_LIGHTS];

vec3 reinhard(vec3 v)
{
    return v / (1.0f + v);
}

float random(vec3 foo, float bar) {
    vec4 seed4 = vec4(foo, bar);
    float dot_product = dot(seed4, vec4(12.9898, 78.233, 45.164, 94.673));
    return fract(sin(dot_product) * 43758.5453);
}

// -- BDRF METHODS -- //

float ggxtr_ndf(vec3 n, vec3 h, float r)
{
    float a2 = r * r * r * r;
    float NH2 = pow(max(0.001f, dot(n, h)), 2);
    return a2 / (M_PI * (pow(NH2 * (a2 - 1.f) + 1.f, 2)));
}

float geo_atten(vec3 wi, vec3 wo, vec3 n, float r)
{
    //float k = (r*r) / 2.f;
    float k = pow((r*r) + 1.f, 2.f) / 8.f;

    float NL = max(dot(n, wi), 0.01f);
    float NV = max(dot(n, wo), 0.01f);

    float G1 = NL / (NL * (1.f - k) + k);
    float G2 = NV / (NV * (1.f - k) + k);

    return G1 * G2;
}

vec3 fresnel(vec3 f0, vec3 h, vec3 wo) {
    return f0 + (1 - f0) * pow(1 - max(0.0, dot(h, wo)), 5);
}

vec3 fresnelRoughness(vec3 f0, vec3 n, vec3 wo, float r)
{
    return f0 + (max(vec3(1.0 - r), f0) - f0) * pow(clamp(1.0 - max(0.0, dot(n, wo)), 0.0, 1.0), 5.0);
}

vec3 Li_point(PointLight light)
{
    //float distance = length(lights[i].position - FragPos);
    //float attenuation = 1.0 / (distance * distance);
    return light.color * light.intensity;// *attenuation;
}

vec3 Li_dir(DirectionalLight light)
{
    return light.color * light.intensity;
}

vec3 f(vec3 wi, vec3 wo, vec3 n, vec3 a, vec3 f0, float r, float m)
{
    vec3 wh = normalize(wo + wi);

    float n_dot_wi = max(dot(n, wi), 0.01f);
    float n_dot_wo = max(dot(n, wo), 0.01f);

    float D = ggxtr_ndf(n, wh, r);
    float G = geo_atten(wi, wo, n, r);
    vec3 F = fresnel(f0, wh, wo);
    vec3 spec = (D * G * F) / max((4.f * n_dot_wo * n_dot_wi), 0.001);

    vec3 kD = vec3(1.0) - F;
    kD *= 1.0 - m;
    vec3 diff = kD * a / M_PI;

    return (diff + spec) * n_dot_wi;
}

void main()
{
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 n = texture(gNormal, TexCoords).rgb;
    vec3 a = texture(gAlbedo, TexCoords).rgb;
    vec3 emissive = texture(gEmissive, TexCoords).rgb;
    float ao = texture(gMetallicRoughAO, TexCoords).r;
    float r = texture(gMetallicRoughAO, TexCoords).g;
    float m = texture(gMetallicRoughAO, TexCoords).b;

    vec3 wo = normalize(camPos - FragPos);

    vec3 L = vec3(0.f);
    vec3 f0 = vec3(0.04);
    f0 = mix(f0, a, m);

    for (int i = 0; i < NR_POINT_LIGHTS; ++i) {
        vec3 wi = normalize(pnt_lights[i].position - FragPos);
        L += f(wi, wo, n, a, f0, r, m) * Li_point(pnt_lights[i]);
    }

    for (int i = 0; i < NR_DIR_LIGHTS; ++i) {
        //vec4 FragPosLightSpace = lightSpaceMatrix[i] * vec4(FragPos, 1.f);
        vec3 wi = normalize(dir_lights[i].direction);
        L += f(wi, wo, n, a, f0, r, m) * Li_dir(dir_lights[i]);// *pcss_shadow(FragPosLightSpace, i);
    }

    vec3 F = fresnelRoughness(f0, n, wo, r);

    vec3 reflection = reflect(-wo, n);
    vec3 prefilteredColor = textureLod(prefilterMap, reflection, r * MAX_REFLECTION_LOD).rgb;

    vec2 envBRDF = texture(brdfLUT, vec2(max(dot(n, wo), 0.0), r)).rg;
    vec3 spec_ambient = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - m;

    vec3 irradiance = texture(irradianceMap, n).rgb;
    vec3 diff_ambient = kD * irradiance * a;

    float shadow = texture(gShadows, TexCoords).r;

    vec3 ambient = (diff_ambient + spec_ambient) * ao;
    vec3 mapped = ((L * (1.f-shadow)) + ambient + emissive) * cam_exposure;

    //vec3 wi = normalize(dir_lights[0].direction);
    //L = f(wi, wo, n, a, f0, r, m);

    //FragColor = vec4(reinhard(ambient * cam_exposure), 1.0);
    FragColor = vec4(reinhard(mapped), 1.0);
}

