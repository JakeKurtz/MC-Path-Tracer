#pragma once
#include <cuda_runtime.h>
#include "Vector.h"

struct dVertex;

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 texCoords;
	glm::vec3 tangent;
	glm::vec3 bitangent;

	__host__ __device__ operator dVertex () const;
	__host__ __device__ Vertex& operator= (const dVertex& v);
};

struct dVertex {
	jek::Vec3f position;
	jek::Vec3f normal;
	jek::Vec2f texcoords;
	jek::Vec3f tangent;
	jek::Vec3f bitangent;
};