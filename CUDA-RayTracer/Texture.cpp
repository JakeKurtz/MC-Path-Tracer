#include "Texture.h"
#include <stb_image.h>

using namespace std;

Texture::Texture() {

}

Texture::Texture(string const& _path, GLenum _target, bool flip, bool mipmap, bool gamma, string _name, GLint _wrap, GLint _mag_filter, GLint _min_filter)
{
    filepath = _path;
    target = _target;
    name = _name;
    wrap = _wrap;
    mag_filter = _mag_filter;
    min_filter = _min_filter;

    glGenTextures(1, &id);
    glBindTexture(target, id);

    if (flip) stbi_set_flip_vertically_on_load(true);

    int data_loaded;
    if (stbi_is_hdr(filepath.c_str())) data_loaded = loadHDR(filepath, target);
    else data_loaded = load(filepath, target);

    if (!data_loaded)
    {
        std::cout << "Texture failed to load at path: " << filepath << std::endl;
    }

    if (target == GL_TEXTURE_1D || target == GL_TEXTURE_2D || target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    if (target == GL_TEXTURE_2D || target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);
    if (target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_R, wrap);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter);

    if (mipmap) glGenerateMipmap(target);

    stbi_set_flip_vertically_on_load(false);
};

Texture::Texture(const aiTexture* tex, GLenum _target, bool flip, bool mipmap, bool gamma, string _name, GLint _wrap, GLint _mag_filter, GLint _min_filter)
{
    filepath = tex->mFilename.C_Str();
    target = _target;
    name = _name;
    wrap = _wrap;
    mag_filter = _mag_filter;
    min_filter = _min_filter;

    glGenTextures(1, &id);
    glBindTexture(target, id);

    if (flip) stbi_set_flip_vertically_on_load(true);

    int data_loaded = loadEmbedded(tex);

    if (!data_loaded)
    {
        std::cout << "Texture failed to load at path: " << _name << std::endl;
    }

    glGenerateMipmap(target);

    if (target == GL_TEXTURE_1D || target == GL_TEXTURE_2D || target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_S, wrap);
    if (target == GL_TEXTURE_2D || target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_T, wrap);
    if (target == GL_TEXTURE_3D) glTexParameteri(target, GL_TEXTURE_WRAP_R, wrap);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, mag_filter);

    if (mipmap) glGenerateMipmap(target);

    stbi_set_flip_vertically_on_load(false);
};

unsigned int Texture::loadHDR(string filename, GLenum target)
{
    float* data = stbi_loadf(filename.c_str(), &width, &height, &nrComponents, 0);

    format = GL_RGB;
    internalformat = GL_RGB16F;
    if (nrComponents == 1) {
        format = GL_RED;
        internalformat = GL_R16;
    }
    else if (nrComponents == 3) {
        format = GL_RGB;
        internalformat = GL_RGB16F;
    }
    else if (nrComponents == 4) {
        format = GL_RGBA;
        internalformat = GL_RGBA16F;
    }

    if (data)
    {
        type = GL_FLOAT;
        if (target == GL_TEXTURE_1D)
            glTexImage1D(target, 0, internalformat, width, 0, format, type, data);
        if (target == GL_TEXTURE_2D)
            glTexImage2D(target, 0, internalformat, width, height, 0, format, type, data);
        if (target == GL_TEXTURE_3D)
            glTexImage3D(target, 0, internalformat, width, height, 1, 0, format, type, data);

        stbi_image_free(data);
        return 1;
    }
    else return 0;
}

unsigned int Texture::load(string filename, GLenum target)
{
    data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);

    format = GL_RED;
    if (nrComponents == 1)
        format = GL_RED;
    else if (nrComponents == 3)
        format = GL_RGB;
    else if (nrComponents == 4)
        format = GL_RGBA;

    if (data)
    {
        type = GL_UNSIGNED_BYTE;
        if (target == GL_TEXTURE_1D)
            glTexImage1D(target, 0, format, width, 0, format, type, data);
        if (target == GL_TEXTURE_2D)
            glTexImage2D(target, 0, format, width, height, 0, format, type, data);
        if (target == GL_TEXTURE_3D)
            glTexImage3D(target, 0, format, width, height, 1, 0, format, type, data);

        //stbi_image_free(data);
        return 1;
    }
    else return 0;
}

unsigned int Texture::loadEmbedded(const aiTexture* tex) {
    if (tex->mHeight == 0) {
        data = stbi_load_from_memory(reinterpret_cast<unsigned char*>(tex->pcData), tex->mWidth, &width, &height, &nrComponents, 0);
    }
    else {
        data = stbi_load_from_memory(reinterpret_cast<unsigned char*>(tex->pcData), tex->mWidth * tex->mHeight, &width, &height, &nrComponents, 0);
    }

    format = GL_RED;
    if (nrComponents == 1)
        format = GL_RED;
    else if (nrComponents == 3)
        format = GL_RGB;
    else if (nrComponents == 4)
        format = GL_RGBA;

    if (data)
    {
        type = GL_UNSIGNED_BYTE;
        if (target == GL_TEXTURE_1D) glTexImage1D(target, 0, format, width, 0, format, type, data);
        if (target == GL_TEXTURE_2D) glTexImage2D(target, 0, format, width, height, 0, format, type, data);
        if (target == GL_TEXTURE_3D) glTexImage3D(target, 0, format, width, height, 1, 0, format, type, data);

        //stbi_image_free(data);
        return 1;
    }
    else return 0;
}