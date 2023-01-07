#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "dMath.cuh"
#include "EnvironmentLight.h"
#include "DirectionalLight.h"

#include <surface_functions.h>

#include "CudaHelpers.h"
#include "BVH.h"

#include <surface_functions.h>
#include "Wavefront.cuh"
#include "Helpers.cuh"
#include "dTexture.cuh"

#include "CudaHelpers.h"

void init_light_on_device(dEnvironmentLight* light);
void build_environment_light(dEnvironmentLight* env_light);
void init_light_on_device(dDirectionalLight* light);