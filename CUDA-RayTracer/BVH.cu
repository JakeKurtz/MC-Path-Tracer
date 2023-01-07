#include "BVH.h"
#include "MemoryArena.h"
#include <Math.h>

bool overlaps(const Bounds3f& b1, const Bounds3f& b2)
{
	bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
	bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
	bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
	return (x && y && z);
}

bool inside(jek::Vec3f& p, const Bounds3f& b)
{
	return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
		p.y >= b.pMin.y && p.y <= b.pMax.y &&
		p.z >= b.pMin.z && p.z <= b.pMax.z);
}

bool inside_exclusive(const jek::Vec3f& p, const Bounds3f& b)
{
	return (p.x >= b.pMin.x && p.x < b.pMax.x&&
		p.y >= b.pMin.y && p.y < b.pMax.y&&
		p.z >= b.pMin.z && p.z < b.pMax.z);
}

Bounds3f Union(const Bounds3f& b, const jek::Vec3f& p)
{
	return Bounds3f(
		jek::Vec3f(
			fmin(b.pMin.x, p.x),
			fmin(b.pMin.y, p.y),
			fmin(b.pMin.z, p.z)),
		jek::Vec3f(
			fmax(b.pMax.x, p.x),
			fmax(b.pMax.y, p.y),
			fmax(b.pMax.z, p.z)));
}

Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2)
{
	return Bounds3f(
		jek::Vec3f(
			fmin(b1.pMin.x, b2.pMin.x),
			fmin(b1.pMin.y, b2.pMin.y),
			fmin(b1.pMin.z, b2.pMin.z)),
		jek::Vec3f(
			fmax(b1.pMax.x, b2.pMax.x),
			fmax(b1.pMax.y, b2.pMax.y),
			fmax(b1.pMax.z, b2.pMax.z)));
}

BVHAccel::BVHAccel(std::vector<BVHPrimitiveInfo> primitiveInfo, SplitMethod splitMethod, int maxPrimsInNode) :
	maxPrimsInNode(maxPrimsInNode), splitMethod(splitMethod)
{
	int totalNodes = 0;

	MemoryArena arena(1024 * 1024);

	if (primitiveInfo.size() > 0) {
		root = recursive_build(arena, primitiveInfo, 0, primitiveInfo.size(), &totalNodes, orderedPrims);
		nodes = AllocAligned<LinearBVHNode>(totalNodes);

		int offset = 0;
		flatten_tree(root, &offset);
	}
}

int BVHAccel::get_nmb_nodes()
{
	return num_nodes;
}

LinearBVHNode* BVHAccel::get_nodes()
{
	return nodes;
}

std::vector<int> BVHAccel::get_ordered_prims()
{
	return orderedPrims;
}

BVHBuildNode* BVHAccel::recursive_build(MemoryArena& arena, std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int* totalNodes, std::vector<int>& orderedPrims)
{
	BVHBuildNode* node = arena.Alloc<BVHBuildNode>();
	(*totalNodes)++;

	Bounds3f bounds;
	for (int i = start; i < end; ++i)
		bounds = Union(bounds, primitiveInfo[i].bounds);

	int nPrimitives = end - start;
	if (nPrimitives == 1) {
		// Create leaf BVHBuildNode //
		int firstPrimOffset = orderedPrims.size();
		for (int i = start; i < end; ++i) {
			int primNum = primitiveInfo[i].primitiveNumber;
			orderedPrims.push_back(primNum);
		}
		node->init_leaf(firstPrimOffset, nPrimitives, bounds);
		return node;
	}
	else {
		// Compute bound of primitive centroids, choose split dimension dim //
		Bounds3f centroidBounds;
		for (int i = start; i < end; ++i)
			centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
		int dim = centroidBounds.maximum_extent();

		// Partition primitives into two sets and build children //
		int mid = (start + end) / 2;

		float con;
		switch (dim) {
		case 0:
			con = centroidBounds.pMax.x == centroidBounds.pMin.x;
			break;
		case 1:
			con = centroidBounds.pMax.y == centroidBounds.pMin.y;
			break;
		default:
			con = centroidBounds.pMax.z == centroidBounds.pMin.z;
		}
		if (con) {
			//if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
				// Create leaf BVHBuildNode //
			int firstPrimOffset = orderedPrims.size();
			for (int i = start; i < end; ++i) {
				int primNum = primitiveInfo[i].primitiveNumber;
				orderedPrims.push_back(primNum);
			}
			node->init_leaf(firstPrimOffset, nPrimitives, bounds);
			return node;
		}
		else {
			// Partition primitives based on splitMethod //
			switch (splitMethod) {
			case SplitMethod::Middle: {
				// Partition primitives through node’s midpoint /
				float pmid;
				switch (dim) {
				case 0:
					pmid = (centroidBounds.pMin.x + centroidBounds.pMax.x) / 2;
					break;
				case 1:
					pmid = (centroidBounds.pMin.y + centroidBounds.pMax.y) / 2;
					break;
				default:
					pmid = (centroidBounds.pMin.z + centroidBounds.pMax.z) / 2;
				}
				//float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) / 2;
				BVHPrimitiveInfo* midPtr = std::partition(
					&primitiveInfo[start], &primitiveInfo[end - 1] + 1,
					[dim, pmid](const BVHPrimitiveInfo& pi) {
						switch (dim) {
						case 0:
							return pi.centroid.x < pmid;
						case 1:
							return pi.centroid.y < pmid;
						default:
							return pi.centroid.z < pmid;
						}
						//return pi.centroid[dim] < pmid;
					});
				mid = midPtr - &primitiveInfo[0];
				if (mid != start && mid != end) break;
			}
			case SplitMethod::EqualCounts: {
				// Partition primitives into equally sized subsets //
				mid = (start + end) / 2;
				std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
					&primitiveInfo[end - 1] + 1,
					[dim](const BVHPrimitiveInfo& a,
						const BVHPrimitiveInfo& b) {
							switch (dim) {
							case 0:
								return a.centroid.x < b.centroid.x;
							case 1:
								return a.centroid.y < b.centroid.y;
							default:
								return a.centroid.z < b.centroid.z;
							}
							//return a.centroid[dim] < b.centroid[dim];
					});

				break;
			}
			case SplitMethod::SAH:
			default: {
				// Partition primitives using approximate SAH //
				if (nPrimitives <= 2) {
					// Partition primitives into equally sized subsets //
					mid = (start + end) / 2;
					std::nth_element(
						&primitiveInfo[start],
						&primitiveInfo[mid],
						&primitiveInfo[end - 1] + 1,
						[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
							switch (dim) {
							case 0:
								return a.centroid.x < b.centroid.x;
							case 1:
								return a.centroid.y < b.centroid.y;
							default:
								return a.centroid.z < b.centroid.z;
							}
							//return a.centroid[dim] < b.centroid[dim];
						});
				}
				else {

					// Allocate BucketInfo for SAH partition buckets //
					constexpr int nBuckets = 12;
					struct BucketInfo {
						int count = 0;
						Bounds3f bounds;
					};
					BucketInfo buckets[nBuckets];

					// Initialize BucketInfo for SAH partition buckets //
					for (int i = start; i < end; ++i) {
						//int b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid)[dim];
						int b;
						switch (dim) {
						case 0:
							b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).x;
							break;
						case 1:
							b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).y;
							break;
						default:
							b = nBuckets * centroidBounds.offset(primitiveInfo[i].centroid).z;
						}
						if (b == nBuckets) b = nBuckets - 1;
						buckets[b].count++;
						buckets[b].bounds = Union(buckets[b].bounds, primitiveInfo[i].bounds);
					}

					// Compute costs for splitting after each bucket //
					float cost[nBuckets - 1];
					for (int i = 0; i < nBuckets - 1; ++i) {
						Bounds3f b0, b1;
						int count0 = 0, count1 = 0;
						for (int j = 0; j <= i; ++j) {
							b0 = Union(b0, buckets[j].bounds);
							count0 += buckets[j].count;
						}
						for (int j = i + 1; j < nBuckets; ++j) {
							b1 = Union(b1, buckets[j].bounds);
							count1 += buckets[j].count;
						}
						cost[i] = .125f + (count0 * b0.surface_area() + count1 * b1.surface_area()) / bounds.surface_area();
					}

					// Find bucket to split at that minimizes SAH metric //
					float minCost = cost[0];
					int minCostSplitBucket = 0;
					for (int i = 1; i < nBuckets - 1; ++i) {
						if (cost[i] < minCost) {
							minCost = cost[i];
							minCostSplitBucket = i;
						}
					}

					// Either create leaf or split primitives at selected SAH bucket //
					float leafCost = nPrimitives;
					if (nPrimitives > maxPrimsInNode || minCost < leafCost) {
						BVHPrimitiveInfo* pmid = std::partition(
							&primitiveInfo[start],
							&primitiveInfo[end - 1] + 1,
							[=](const BVHPrimitiveInfo& pi) {
								//int b = nBuckets * centroidBounds.offset(pi.centroid)[dim];
								int b;
								switch (dim) {
								case 0:
									b = nBuckets * centroidBounds.offset(pi.centroid).x;
									break;
								case 1:
									b = nBuckets * centroidBounds.offset(pi.centroid).y;
									break;
								default:
									b = nBuckets * centroidBounds.offset(pi.centroid).z;
								}
								if (b == nBuckets) b = nBuckets - 1;
								return b <= minCostSplitBucket;
							});
						mid = pmid - &primitiveInfo[0];
					}
					else {
						// Create leaf BVHBuildNode //
						int firstPrimOffset = orderedPrims.size();
						for (int i = start; i < end; ++i) {
							int primNum = primitiveInfo[i].primitiveNumber;
							orderedPrims.push_back(primNum);
						}
						node->init_leaf(firstPrimOffset, nPrimitives, bounds);
						return node;
					}
				}
				break;
			}
			}
			auto child0 = recursive_build(arena, primitiveInfo, start, mid, totalNodes, orderedPrims);
			auto child1 = recursive_build(arena, primitiveInfo, mid, end, totalNodes, orderedPrims);
			node->init_interior(dim, child0, child1);
		}
	}
	return node;
}

int BVHAccel::flatten_tree(BVHBuildNode* node, int* offset)
{
	LinearBVHNode* linearNode = &nodes[*offset];
	num_nodes++;
	linearNode->bounds = node->bounds;

	int myOffset = (*offset)++;
	if (node->nPrimitives > 0)
	{
		linearNode->primitivesOffset = node->firstPrimOffset;
		linearNode->nPrimitives = node->nPrimitives;
	}
	else
	{
		// Create interior flattened BVH node //
		linearNode->axis = node->splitAxis;
		linearNode->nPrimitives = 0;

		flatten_tree(node->children[0], offset);
		linearNode->secondChildOffset = flatten_tree(node->children[1], offset);
	}
	return myOffset;
}
