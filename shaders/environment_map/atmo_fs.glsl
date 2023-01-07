#version 330 core
out vec4 FragColor;

in vec3 WorldPos;

uniform vec3 camPos;
uniform vec3 camDir;
uniform vec3 camUp;
uniform vec3 camRight;

uniform vec2 iResolution;

// Atmosphere Properties //

const int MAX_VIEW_SAMPLES = 64;
const int MAX_LIGHT_SAMPLES = 4;

uniform float scale_height;
uniform float planetRadius;

uniform float scaleHeight_rayleigh;
uniform float scaleHeight_mie;

uniform float ray_intensity;
uniform float mie_intensity;
uniform float absorption_intensity;

uniform vec3 planetCenter;
uniform float atmosphereRadius;

uniform float ap_world_intensity;
uniform float ap_cloud_intensity;

// Light Properties //

uniform vec3 lightColor;
uniform float lightIntensity;
uniform vec3 lightDir;

#define PI 3.14159265358979323846264338327

// Scattering coefficients
vec3 beta_ray = vec3(5.19673e-6, 12.1427e-6, 29.6453e-6) * ray_intensity;
vec3 beta_mie = vec3(21e-6) * mie_intensity;
vec3 beta_ozone = vec3(2.04e-5, 4.97e-5, 1.95e-6) * absorption_intensity;

float bn;
float worldDepth;
float bar;
vec3 atmoColor = vec3(0.099, 0.2, 0.32);

// ------------------------------------------------------------ //
//                      General Methods                         //
// ------------------------------------------------------------ //

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}

float remap(float x, float low1, float high1, float low2, float high2) {
    return low2 + (x - low1) * (high2 - low2) / (high1 - low1);
}

float linearize_depth(float d, float zNear, float zFar) {
    return zNear * zFar / (zFar + d * (zNear - zFar));
}

vec3 luminance(vec3 col) {
    float R = col.r;
    float G = col.g;
    float B = col.b;
    return vec3(0.299 * R + 0.587 * G + 0.114 * B);
}

// ------------------------------------------------------------ //
//                      Phase Functions                         //
// ------------------------------------------------------------ //

float HG_Phase(float g, vec3 lightDir, vec3 viewDir) {
    float theta = dot(lightDir, viewDir);
    float g2 = g * g;
    return 1.f / (4.f * PI) * ((1.f - g2) / pow((1.f + g2 - (2 * g * theta)), 1.5f));
}

float Rayleigh_Phase(vec3 lightDir, vec3 viewDir) {
    float theta = dot(lightDir, viewDir);
    return 3 / (16 * PI) * (1.0 + (theta * theta));
}

// ------------------------------------------------------------ //
//                  Sphere Intersection Methods                 //
// ------------------------------------------------------------ //

void solveQuadratic(float a, float b, float c, float d, out float t0, out float t1) {
    if (d > 0.0) {
        t0 = max((-b - sqrt(d)) / (2.0 * a), 0.0);
        t1 = (-b + sqrt(d)) / (2.0 * a);
        return;
    }
    else {
        t0 = 1e32;
        t1 = 0;
        return;
    }
}

void sphereIntersect(vec3 center, float radius, vec3 rayOrigin, vec3 rayDir, out float t0, out float t1) {
    vec3 L = rayOrigin - center;

    float a = dot(rayDir, rayDir);
    float b = 2.0 * dot(rayDir, L);
    float c = dot(L, L) - (radius * radius);
    float d = (b * b) - 4.0 * a * c;

    solveQuadratic(a, b, c, d, t0, t1);
}

void atmoIntersection(vec3 rayPos, vec3 rayDir, out float x, out float y) {

    float t0, t1;
    sphereIntersect(planetCenter, atmosphereRadius, rayPos, rayDir, t0, t1);
    if (t1 > 0) t1 -= t0;

    float s0, s1;
    sphereIntersect(planetCenter, planetRadius, rayPos, rayDir, s0, s1);
    if (s1 > 0) s1 -= s0;

    if (s1 < 0) {
        s0 = 1e32;
        s1 = 0;
    }

    x = min(t1, s0 - t0);
    y = t0;
}

// ------------------------------------------------------------ //
//              Volumetric Atmosphere Methods                   //
// ------------------------------------------------------------ //

float atmoDensity(vec3 pos, float scaleHeight) {
    float h = remap(pos.y, planetRadius, atmosphereRadius, 0.f, 1.f);
    return exp(-(h / scaleHeight));
}

void atmoRayLight(vec3 rayOrigin, vec3 rayDir, float rayLength, out float lightOpticalDepth_ray, out float lightOpticalDepth_mie) {
    float marchPos = 0.0;
    float stepSize = rayLength / float(MAX_LIGHT_SAMPLES);

    rayOrigin += (bn * stepSize * 0.01);

    lightOpticalDepth_ray = 0.f;
    lightOpticalDepth_mie = 0.f;

    for (int i = 0; i < MAX_LIGHT_SAMPLES; i++) {
        vec3 densitySamplePoint = rayOrigin + rayDir * (marchPos + 0.5 * stepSize);

        float density_ray = atmoDensity(densitySamplePoint, scaleHeight_rayleigh) * stepSize;
        float density_mie = atmoDensity(densitySamplePoint, scaleHeight_mie) * stepSize;

        lightOpticalDepth_ray += density_ray;
        lightOpticalDepth_mie += density_mie;

        marchPos += stepSize;
    }
}

vec3 getTransmittance(vec3 currentPos, vec3 lightDir, float viewOpticalDepth_ray, float viewOpticalDepth_mie) {

    float t0, lightRayLength;
    sphereIntersect(planetCenter, atmosphereRadius, currentPos, lightDir, t0, lightRayLength);

    if (lightRayLength > 0) {
        lightRayLength -= t0;
    }

    float lightOpticalDepth_ray, lightOpticalDepth_mie;
    atmoRayLight(currentPos, lightDir, lightRayLength, lightOpticalDepth_ray, lightOpticalDepth_mie);

    return exp(-(beta_ray * (lightOpticalDepth_ray + viewOpticalDepth_ray) +
        beta_mie * (lightOpticalDepth_mie + viewOpticalDepth_mie) +
        beta_ozone * (lightOpticalDepth_ray + viewOpticalDepth_ray)));
}

vec3 atmoRayMarch(vec3 rayOrigin, vec3 rayDir, float rayLength, out vec3 opacity) {

    float stepSize = rayLength / float(MAX_VIEW_SAMPLES);
    float marchPos = 0.0;

    float phase_ray = Rayleigh_Phase(lightDir, rayDir);
    float phase_mie = HG_Phase(0.99, lightDir, rayDir);

    float viewOpticalDepth_ray = 0.0;
    float viewOpticalDepth_mie = 0.0;

    vec3 inScatter_ray = vec3(0);
    vec3 inScatter_mie = vec3(0);

    for (int i = 0; i < MAX_VIEW_SAMPLES; ++i) {

        vec3 currentPos = rayOrigin + rayDir * (marchPos + 0.5 * stepSize);

        float density_ray = atmoDensity(currentPos, scaleHeight_rayleigh) * stepSize;
        float density_mie = atmoDensity(currentPos, scaleHeight_mie) * stepSize;

        vec3 transmittance = getTransmittance(currentPos, lightDir, viewOpticalDepth_ray, viewOpticalDepth_mie);

        viewOpticalDepth_ray += density_ray;
        viewOpticalDepth_mie += density_mie;

        inScatter_ray += density_ray * transmittance;
        inScatter_mie += density_mie * transmittance;

        marchPos += stepSize;
    }

    opacity = exp(-(beta_mie * viewOpticalDepth_mie + beta_ray * viewOpticalDepth_ray + beta_ozone * viewOpticalDepth_ray));

    return ((inScatter_ray * beta_ray * phase_ray) + (inScatter_mie * beta_mie * phase_mie)) * (lightIntensity * lightColor) + vec3(0.f) * opacity;
}

void main() 
{
    vec2 uv = (gl_FragCoord.xy / iResolution.xy) - vec2(0.5);
    uv.x *= iResolution.x / iResolution.y;

    float fov = 90;
    float Px = (2.f * ((gl_FragCoord.x + 0.5) / iResolution.x) - 1.f) * tan(fov / 2.f * PI / 180.f) * (iResolution.x / iResolution.y);
    float Py = (1.f - 2.f * (gl_FragCoord.y + 0.5) / iResolution.y) * tan(fov / 2.f * PI / 180.f);

    uv = vec2(Px, Py);

    vec3 rayDir = mat3(camRight, camUp, camDir) * normalize(vec3(uv, 1.0));
    vec3 rayPos = camPos;

    bn = 0.0;//texture(blueNoise, uv * 50).r;
    //worldDepth = //texture(cameraDepthTexture, gl_FragCoord.xy / iResolution.xy).r;
    worldDepth = 0.f;//linearize_depth(worldDepth, 0.1f, 1000000.f);

    float dstInsideAtmo, dstToAtmo;
    atmoIntersection(rayPos, rayDir, dstInsideAtmo, dstToAtmo);

    // Render atmosphere
    float atmoRayLength = dstInsideAtmo;//min(worldDepth, dstInsideAtmo);

    vec3 atmoOpacity = vec3(1.f);
    if (atmoRayLength > 0) {
        vec3 pointInAtmo = rayPos + rayDir * dstToAtmo;
        atmoColor = 1.0 - exp(-atmoRayMarch(pointInAtmo, rayDir, atmoRayLength, atmoOpacity));
    }

    // Render sun
    vec3 sunPos = vec3(0.f) + lightDir* 1000000.f;
    float sunRadius = 10000.f;
    float t0, t1;
    sphereIntersect(sunPos, sunRadius, rayPos, rayDir, t0, t1);

    if ((t0 > 0 && t1 > 0) || t0 == t1) { // we hit the SUN of a bitch!
        atmoColor *= 12000.f;
    }

    FragColor = vec4(atmoColor, 1.f);
}