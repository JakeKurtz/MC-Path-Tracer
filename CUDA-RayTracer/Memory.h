#pragma once
#include <corecrt_malloc.h>

#define ALLOCA(TYPE, COUNT) (TYPE *)alloca((COUNT) * sizeof(TYPE))

#ifndef PBRT_L1_CACHE_LINE_SIZE
#define PBRT_L1_CACHE_LINE_SIZE 64
#endif

void* AllocAligned(size_t size) {
#if defined(PBRT_IS_WINDOWS)
    return _aligned_malloc(size, PBRT_L1_CACHE_LINE_SIZE);
#elif defined (PBRT_IS_OPENBSD) || defined(PBRT_IS_OSX)
    void* ptr;
    if (posix_memalign(&ptr, PBRT_L1_CACHE_LINE_SIZE, size) != 0)
        ptr = nullptr;
    return ptr;
#else
    return _aligned_malloc(size, PBRT_L1_CACHE_LINE_SIZE);
    //return aligned_alloc(PBRT_L1_CACHE_LINE_SIZE, size);
#endif
}

template <typename T> T* AllocAligned(size_t count) {
    return (T*)AllocAligned(count * sizeof(T));
}

void FreeAligned(void* ptr) {
    if (!ptr) return;
#if defined(PBRT_HAVE__ALIGNED_MALLOC)
    _aligned_free(ptr);
#else
    _aligned_free(ptr);//free(ptr);
#endif
}