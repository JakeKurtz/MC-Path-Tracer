#pragma once

#include <vector>
#include <algorithm>
#include <vector_types.h>
#include <vector_functions.h>
#include "Bounds3f.h"

class MemoryArena;

const int MAX_LEAF = 100;

__device__ __host__ bool overlaps(const Bounds3f& b1, const Bounds3f& b2);

__device__ __host__ bool inside(jek::Vec3f& p, const Bounds3f& b);

__device__ __host__ bool inside_exclusive(const jek::Vec3f& p, const Bounds3f& b);

__device__ __host__ Bounds3f Union(const Bounds3f& b, const jek::Vec3f& p);

__device__ __host__ Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2);

enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

struct BVHPrimitiveInfo
{
	__device__ __host__ BVHPrimitiveInfo() {}

	__device__ __host__ BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f& bounds) :
		primitiveNumber(primitiveNumber), bounds(bounds),
		centroid(0.5f * bounds.pMin + 0.5f * bounds.pMax)
	{}
	size_t primitiveNumber;
	Bounds3f bounds;
	jek::Vec3f centroid;
};

struct BVHBuildNode
{
	Bounds3f bounds;
	BVHBuildNode* children[2];
	size_t primitiveNumber;
	int splitAxis, firstPrimOffset, nPrimitives, obj_id;

	void init_leaf(int first, int n, const Bounds3f& b)
	{
		firstPrimOffset = first;
		nPrimitives = n;
		bounds = b;
		children[0] = children[1] = nullptr;
	}

	void init_interior(int axis, BVHBuildNode* c0, BVHBuildNode* c1)
	{
		children[0] = c0;
		children[1] = c1;
		splitAxis = axis;
		bounds = Union(c0->bounds, c1->bounds);
		nPrimitives = 0;
	}
};

struct LinearBVHNode {
	Bounds3f bounds;
	union {
		int primitivesOffset;    // leaf
		int secondChildOffset;   // interior
	};
	uint16_t nPrimitives;  // 0 -> interior node
	uint8_t axis;          // interior node: xyz
	uint8_t pad[1];        // ensure 32 byte total size
};

class BVHAccel {
public:
	BVHAccel(std::vector<BVHPrimitiveInfo> primitiveInfo, SplitMethod splitMethod, int maxPrimsInNode);
	int get_nmb_nodes();
	LinearBVHNode* get_nodes();
	std::vector<int> get_ordered_prims();
private:

	BVHBuildNode* recursive_build(MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int* totalNodes, std::vector<int>& orderedPrims);
	int flatten_tree(BVHBuildNode* node, int* offset);

	std::vector<int> orderedPrims;
	const int maxPrimsInNode;
	const SplitMethod splitMethod;
	BVHBuildNode* root;
	LinearBVHNode* nodes = nullptr;
	int num_nodes = 0;
};