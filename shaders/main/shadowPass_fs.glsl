#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform float lightSize;
uniform float searchAreaSize;
uniform float shadowScaler;
uniform int kernel_size;

uniform sampler2D gPosition;
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

float random(vec3 foo, float bar) {
    vec4 seed4 = vec4(foo, bar);
    float dot_product = dot(seed4, vec4(12.9898, 78.233, 45.164, 94.673));
    return fract(sin(dot_product) * 43758.5453);
}

// -- SHADOW METHODS -- //

float pcf_shadow(vec4 fragPosLightSpace, int index)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    float closestDepth = texture(shadowMaps, vec3(projCoords.xy, index)).r;
    float currentDepth = projCoords.z;

    float bias = 0.00005;

    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMaps, 0).xy;
    for (int x = -3; x <= 3; ++x)
    {
        for (int y = -3; y <= 3; ++y)
        {
            vec2 offset = vec2(x, y);//vec2(random(gl_FragCoord.xyy, y), random(gl_FragCoord.xyy, x)) * vec2(x,y);
            float pcfDepth = texture(shadowMaps, vec3(projCoords.xy + offset * texelSize, index)).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= 49.f;

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if (projCoords.z > 1.0)
        shadow = 0.0;

    return 1.f - shadow;
}

float pcf(vec3 projCoords, int index, float penumbraSize)
{
    float closestDepth = texture(shadowMaps, vec3(projCoords.xy, index)).r;
    float currentDepth = projCoords.z;

    float bias = 0.00005;

    // PCF
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMaps, 0).xy;
    int step_count = kernel_size / 2;
    for (int x = -step_count; x <= step_count; ++x)
    {
        for (int y = -step_count; y <= step_count; ++y)
        {
            float offset = 1.f;//random(gl_FragCoord.xyy, y+x);
            float pcfDepth = texture(shadowMaps, vec3(projCoords.xy + vec2(x, y) * texelSize * offset * penumbraSize, index)).r;
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;
        }
    }
    shadow /= (kernel_size * kernel_size);

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if (projCoords.z > 1.0)
        shadow = 0.0;

    return shadow;
}

float pcss_estimatePenumbraSize(float lightSize, float averageBlockerDepth, float receiverDepth)
{
    return lightSize * (receiverDepth - averageBlockerDepth) / averageBlockerDepth;
}

float pcss_blockerDistance(vec3 projCoord, float searchUV, int index)
{
    // Perform N samples with pre-defined offset and random rotation, scale by input search size
    float blockers = 0.f;
    float averageBlockerDepth = 0.0f;

    vec2 texelSize = 1.0 / textureSize(shadowMaps, 0).xy;
    int step_count = 2;//kernel_size / 2;

    for (int x = -step_count; x <= step_count; x++)
    {
        for (int y = -step_count; y <= step_count; ++y)
        {
            // Calculate sample offset (technically anything can be used here - standard NxN kernel, random samples with scale, etc.)
            //vec2 offset = PCSS_Samples[i] * searchUV;
            //offset = PCSS_Rotate(offset, rotationTrig);

            // This is just a standard 5x5 kernel, scaled by the searchUV.
            vec2 offset = vec2(x, y) * texelSize * searchUV * random(gl_FragCoord.xyy, y + x);;

            // Compare given sample depth with receiver depth, if it puts receiver into shadow, this sample is a blocker
            float z = texture(shadowMaps, vec3(projCoord.xy + offset, index)).r;
            if (z < projCoord.z)
            {
                blockers++;
                averageBlockerDepth += z;
            }
        }
    }

    // Calculate average blocker depth
    if (blockers == 0.f) {
        return -1;
    }
    else {
        return averageBlockerDepth / blockers;
    }
}

float pcss_shadow(vec4 fragPosLightSpace, int index)
{
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    float searchUV = searchAreaSize;
    float averageBlockerDepth = pcss_blockerDistance(projCoords, searchUV, index);

    // If there isn't any average blocker distance - it means that there is no blocker at all
    if (averageBlockerDepth == -1.0)
    {
        return 0.f;
    }
    else
    {
        //float lightSize = filterSize;
        float penumbraSize = pcss_estimatePenumbraSize(lightSize, averageBlockerDepth, projCoords.z);
        float shadow = pcf(projCoords, index, penumbraSize);
        return shadow;
    }
}

void main()
{
    vec3 FragPos = texture(gPosition, TexCoords).rgb;

    float shadow = 0.f;

    for (int i = 0; i < 2; ++i) {
        vec3 wi = normalize(pnt_lights[i].position - FragPos);
        //shadow += 0.f;//f(wi, wo, n, a, f0, r, m) * Li_point(pnt_lights[i]);
    }

    for (int i = 0; i < 4; ++i) {
        vec4 FragPosLightSpace = lightSpaceMatrix[i] * vec4(FragPos, 1.f);
        shadow += pcss_shadow(FragPosLightSpace, i) * (shadowScaler*min(dir_lights[i].intensity, 20.f));
    }

    FragColor = vec4(vec3(shadow), 1.f);
    //gShadows = texture(shadowMaps, vec3(TexCoords, 0)).r;//gShadows = shadow;
    //vec4 FragPosLightSpace = lightSpaceMatrix[0] * vec4(FragPos, 1.f);
    //FragColor = vec4(vec3(pcss_shadow(FragPosLightSpace, 0)), 1.f);
}

