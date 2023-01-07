#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Triangle.h"
#include "CudaHelpers.h"
#include "BVH.h"
#include <surface_functions.h>
#include "Wavefront.cuh"
#include "Helpers.cuh"
#include "dTexture.cuh"
#include <cooperative_groups.h>
#include "Film.h"
#include "Camera.h"
#include "Scene.h"

void clear_dfilm(dFilm* film);

void wavefront_init(
    Paths* paths, 
    uint32_t nmb_paths);

void wavefront_pathtrace(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<Film> film,
    Queues* queues,
    cudaArray_const_t array,
    cudaEvent_t event,
    cudaStream_t stream);

void debug_raytracer(
    std::shared_ptr<Scene> s,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<Film> film,
    cudaArray_const_t array,
    cudaEvent_t event,
    cudaStream_t stream);