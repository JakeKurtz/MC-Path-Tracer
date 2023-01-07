#include "light_initialization_kernels.cuh"

__global__
void g_compute_pdf_denom(dEnvironmentLight* env_light)
{
    unsigned int width = env_light->tex_width;
    unsigned int height = env_light->tex_height;

    unsigned int hrd_texture = env_light->hrd_texture;

    float _pdf_denom = 0.0;

    for (int j = 0; j < height; j++)
    {
        float v = (float)j / (float)height;
        float sin_theta = sin(jek::M_PI * v);

        for (int i = 0; i < width; i++)
        {
            float u = (float)i / (float)width;
            float lum = jek::luminance(tex2DLod<float4>(hrd_texture, u, v, 0));
            _pdf_denom += lum * sin_theta;
        }
    }
    env_light->pdf_denom = _pdf_denom;
}
__global__
void g_compute_marginal_dist(dEnvironmentLight* env_light)
{
    unsigned int width = env_light->tex_width;
    unsigned int height = env_light->tex_height;

    unsigned int hrd_texture = env_light->hrd_texture;
    float pdf_denom = env_light->pdf_denom;

    float* marginal_y = env_light->marginal_y;
    float* marginal_p = env_light->marginal_p;

    for (int j = 0; j < height; j++)
    {
        float v = (float)j / (float)height;
        double sin_theta = sin(jek::M_PI * v) / pdf_denom;

        marginal_p[j] = 0.f;
        for (int i = 0; i < width; i++)
        {
            float u = (float)i / (float)width;
            double lum = jek::luminance(tex2DLod<float4>(hrd_texture, u, v, 0));
            marginal_p[j] += lum * sin_theta;
        }

        if (j != 0) marginal_y[j] = marginal_p[j] + marginal_y[j - 1];
        else marginal_y[j] = marginal_p[j];
    }
}
__global__
void g_compute_conditional_dist(dEnvironmentLight* env_light)
{
    unsigned int width = env_light->tex_width;
    unsigned int height = env_light->tex_height;

    unsigned int hrd_texture = env_light->hrd_texture;

    float pdf_denom = env_light->pdf_denom;
    float* conds_y = env_light->conds_y;
    int conds_y_pitch = env_light->conds_y_pitch;
    float* marginal_p = env_light->marginal_p;

    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;
    int y = id % height;

    float v = (float)y / (float)height;
    float sin_theta = sin(jek::M_PI * v);
    float val = sin_theta / (pdf_denom * marginal_p[y]);

    for (int x = 0; x < width; x++) {
        float u = (float)x / (float)width;
        float lum = jek::luminance(tex2DLod<float4>(hrd_texture, u, v, 0));

        float* _conds_y = (float*)((char*)conds_y + y * conds_y_pitch);
        _conds_y[x] = lum * val;
        if (x != 0) _conds_y[x] += _conds_y[x - 1];
    }
}
__global__
void g_write_pdf_texture(dEnvironmentLight* env_light)
{
    uint32_t id = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned int width = env_light->tex_width;
    unsigned int height = env_light->tex_height;

    unsigned int hrd_texture = env_light->hrd_texture;
    unsigned int pdf_texture = env_light->pdf_texture;

    float pdf_denom = env_light->pdf_denom;

    int x = id % width;
    int y = (id / width) % height;

    float u = (float)x / (float)width;
    float v = (float)y / (float)height;

    float sin_theta = sin(jek::M_PI * v);

    float lum = jek::luminance(tex2DLod<float4>(hrd_texture, u, v, 0));

    float pdf = (lum * sin_theta) / pdf_denom;
    //pdf *= (width * height) / (2.f * sin_theta * jek::M_PI * jek::M_PI);

    surf2Dwrite(pdf, pdf_texture, x * sizeof(float), y, cudaBoundaryModeZero);
}
__global__
void g_test(dEnvironmentLight* env_light)
{
    unsigned int width = env_light->tex_width;
    unsigned int height = env_light->tex_height;

    unsigned int hrd_texture = env_light->hrd_texture;
    unsigned int pdf_texture = env_light->pdf_texture;

    float total = 0.f;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            float pdf;
            surf2Dread(&pdf, pdf_texture, i * sizeof(float), j);
            total += pdf;
        }
    }

    printf("%f\n", total);
}
void build_environment_light(dEnvironmentLight* env_light)
{
    auto tex_width = env_light->tex_width;
    auto tex_height = env_light->tex_height;

    int block_size = 256;
    int num_blocks;

    if ((int)env_light->hrd_texture != -1) {
        g_compute_pdf_denom << < 1, 1 >> > (env_light);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        g_compute_marginal_dist << < 1, 1 >> > (env_light);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        num_blocks = (tex_height + block_size - 1) / block_size;
        g_compute_conditional_dist << < num_blocks, block_size >> > (env_light);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        num_blocks = (tex_width * tex_height + block_size - 1) / block_size;
        g_write_pdf_texture << < num_blocks, block_size >> > (env_light);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        g_test << < 1, 1 >> > (env_light);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

__global__
void g_init_light_on_device(dEnvironmentLight* env_light)
{
    new (env_light) dEnvironmentLight();
}
void init_light_on_device(dEnvironmentLight* light)
{
    g_init_light_on_device <<< 1, 1 >>> (light);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void g_init_light_on_device(dDirectionalLight* dir_light)
{
    new (dir_light) dDirectionalLight();
}
void init_light_on_device(dDirectionalLight* light)
{
    g_init_light_on_device << < 1, 1 >> > (light);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
