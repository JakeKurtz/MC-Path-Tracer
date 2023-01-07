/*
 Note to future self.

 This code is really bad. The only "Good" function here is load_texture_uchar. You need to ensure
 that data is properly cleaned up after use. Also, ensure that the mipmappedArray is cleaned up before returning.

 The best thing to do here is to make a class that handels all this bullshit. I'm just too lazy to do that atm.

*/

#include "dTexture.h"
#include "Texture.h"
#include "CudaHelpers.h"
#include <stb_image.h>

uint32_t get_mip_map_levels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width, size.height), size.depth);

    uint32_t levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}

cudaTextureObject_t load_texture_uchar(Texture* tex)
{
    // TODO: make a dTexture class. This code is not great :( Need a better way of handling different texture formats/internal formats/dimensions.
    if (tex != NULL) {
        cudaTextureObject_t textureObject;
        cudaMipmappedArray_t mipmapArray;

        auto size = make_cudaExtent(tex->width, tex->height, 0);
        uint32_t levels = get_mip_map_levels(size);

        cudaChannelFormatDesc desc;
        size_t pitch;

        if (tex->nrComponents == 3) {
            pitch = size.width * sizeof(uchar4);
            desc = cudaCreateChannelDesc<uchar4>();
            checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, levels));

            unsigned char* data = (unsigned char*)malloc(tex->width * tex->height * sizeof(uchar4));

            int i = 0;
            int j = 0;
            while (i < tex->width * tex->height * 4) {
                data[i] = tex->data[j];
                data[i + 1] = tex->data[j + 1];
                data[i + 2] = tex->data[j + 2];
                data[i + 3] = 0;

                i += 4;
                j += 3;
            }
            cudaArray_t level0;
            checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

            cudaMemcpy3DParms copyParams = { 0 };
            copyParams.srcPtr = make_cudaPitchedPtr(data, pitch, size.width, size.height);
            copyParams.dstArray = level0;
            copyParams.extent = size;
            copyParams.extent.depth = 1;
            copyParams.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy3D(&copyParams));

            free(data);
        }
        else {
            
            pitch = size.width * sizeof(uchar4);
            desc = cudaCreateChannelDesc<uchar4>();
            checkCudaErrors(cudaMallocMipmappedArray(&mipmapArray, &desc, size, levels));
            
            cudaArray_t level0;
            checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, mipmapArray, 0));

            cudaMemcpy3DParms copyParams = { 0 };
            copyParams.srcPtr = make_cudaPitchedPtr(tex->data, pitch, size.width, size.height);
            copyParams.dstArray = level0;
            copyParams.extent = size;
            copyParams.extent.depth = 1;
            copyParams.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaMemcpy3D(&copyParams));
        }

        // compute rest of mipmaps based on level 0
        //generateMipMaps(mipmapArray, size);

        // generate bindless texture object
        
        cudaResourceDesc resDescr;
        memset(&resDescr, 0, sizeof(cudaResourceDesc));

        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = mipmapArray;

        cudaTextureDesc texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.mipmapFilterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.addressMode[2] = cudaAddressModeWrap;

        texDescr.maxMipmapLevelClamp = float(levels - 1);

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&textureObject, &resDescr, &texDescr, NULL));
        //checkCudaErrors(cudaFreeMipmappedArray(mipmapArray));

        return textureObject;
    }
    else {
        return -1;
    }
}

cudaTextureObject_t load_texture_uchar(std::string filename)
{
    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* hData = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
    stbi_set_flip_vertically_on_load(false);
    unsigned int size = width * height * sizeof(float4);

    if (hData != nullptr) {

        cudaChannelFormatDesc desc;
        cudaArray* cuArray;

        desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindFloat);

        checkCudaErrors(cudaMallocArray(&cuArray,
            &desc,
            width,
            height));

        if (nrComponents == 3) {

            unsigned int* data = (unsigned int*)malloc(size);

            int i = 0;
            int j = 0;
            while (i < width * height * 4) {
                data[i] = hData[j];
                data[i + 1] = hData[j + 1];
                data[i + 2] = hData[j + 2];
                data[i + 3] = 0;

                i += 4;
                j += 3;
            }
            checkCudaErrors(cudaMemcpyToArray(cuArray,
                0,
                0,
                data,
                size,
                cudaMemcpyHostToDevice));
        }
        else {
            checkCudaErrors(cudaMemcpyToArray(cuArray,
                0,
                0,
                hData,
                size,
                cudaMemcpyHostToDevice));
        }

        cudaTextureObject_t         tex;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = cuArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
        return tex;
    }
    else {
        std::cerr << "failed to load the texture \"" << filename << "\"." << endl;
        return -1;
    }
}

cudaTextureObject_t load_texture_float(std::string filename)
{
    int width, height, nrComponents;

    //stbi_set_flip_vertically_on_load(true);
    float* hData = stbi_loadf(filename.c_str(), &width, &height, &nrComponents, 0);

    unsigned int size = width * height * sizeof(float4);

    if (hData != nullptr) {

        cudaChannelFormatDesc desc;
        cudaArray* cuArray;

        desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        checkCudaErrors(cudaMallocArray(&cuArray,
            &desc,
            width,
            height));

        if (nrComponents == 3) {

            float* data = (float*)malloc(size);

            int i = 0;
            int j = 0;
            while (i < width * height * 4) {
                data[i] = hData[j];
                data[i + 1] = hData[j + 1];
                data[i + 2] = hData[j + 2];
                data[i + 3] = 0;

                i += 4;
                j += 3;
            }
            checkCudaErrors(cudaMemcpyToArray(cuArray,
                0,
                0,
                data,
                size,
                cudaMemcpyHostToDevice));
        }
        else {
            checkCudaErrors(cudaMemcpyToArray(cuArray,
                0,
                0,
                hData,
                size,
                cudaMemcpyHostToDevice));
        }

        cudaTextureObject_t         tex;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = cuArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = true;
        texDescr.filterMode = cudaFilterModeLinear;
        texDescr.addressMode[0] = cudaAddressModeWrap;
        texDescr.addressMode[1] = cudaAddressModeWrap;
        texDescr.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

        stbi_set_flip_vertically_on_load(false);

        return tex;
    }
    else {
        std::cerr << "failed to load the texture \"" << filename << "\"." << endl;
        return -1;
    }
}

cudaSurfaceObject_t create_surface_float(int width, int height, int nrComponents)
{
    unsigned int size;
    cudaChannelFormatDesc desc;
    cudaArray* cuArray;

    switch (nrComponents) {
    case 1:
        size = width * height * sizeof(float);
        desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        break;
    case 2:
        size = width * height * sizeof(float2);
        desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
        break;
    default:
        size = width * height * sizeof(float4);
        desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    }

    checkCudaErrors(cudaMallocArray(&cuArray,
        &desc,
        width,
        height));

    cudaSurfaceObject_t         surf;
    cudaResourceDesc            surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));

    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = cuArray;

    checkCudaErrors(cudaCreateSurfaceObject(&surf, &surfRes));
    return surf;
}

void destroy_texture(cudaTextureObject_t texture) {
    //cuTexObjectDestroy(texture);
    if (texture != -1) cudaDestroyTextureObject(texture);
}
void destroy_surface(cudaSurfaceObject_t surface) {
    //cuTexObjectDestroy(texture);
    if (surface != -1) cudaDestroySurfaceObject(surface);
}