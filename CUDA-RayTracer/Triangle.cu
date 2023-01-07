#include "Triangle.h"
#include "Isect.cuh"
#include "CudaHelpers.h"
#include "BVH.h"

// NOTE: Turning this off will result in artifacts. Needs to be fixed!
#define TEST_CULL

bool dTriangle::intersect(const dRay& ray, float& u, float& v, float& t) const
{
    float _u, _v, _t;

    jek::Vec3f e1 = v1.position - v0.position;
    jek::Vec3f e2 = v2.position - v0.position;

    jek::Vec3f pvec = jek::cross(ray.d, e2);
    double det = jek::dot(e1, pvec);

#ifdef TEST_CULL
    if (det < jek::K_EPSILON)
        return false;

    jek::Vec3f tvec = ray.o - v0.position;
    _u = jek::dot(tvec, pvec);
    if (_u < 0.0 || _u > det)
        return false;

    jek::Vec3f qvec = jek::cross(tvec, e1);
    _v = jek::dot(ray.d, qvec);
    if (_v < 0.0 || _u + _v > det)
        return false;

    _t = jek::dot(e2, qvec);

    double inv_det = 1.0 / det;

    _u *= inv_det;
    _v *= inv_det;
    _t *= inv_det;

#else
    if (det > -jek::K_EPSILON && det < jek::K_EPSILON)
        return false;

    double inv_det = 1.0 / det;

    jek::Vec3f tvec = ray.o - v0.position;
    u = jek::dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    jek::Vec3f qvec = jek::cross(tvec, e1);
    v = jek::dot(ray.d, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    t = jek::dot(e2, qvec) * inv_det;
#endif

    u = _u;
    v = _v;
    t = _t;

    return true;
}
bool dTriangle::hit(const dRay& ray, float& tmin, Isect& isect) const
{
    // apply inverse transformation matrix to ray
    dRay ray_p = transform->inv_matrix * ray;

    float u, v, t;
    bool hit = intersect(ray_p, u, v, t);

    if (t < 0 || !hit) return false;

    jek::Vec3f normal = jek::normalize(u * v1.normal + v * v2.normal + (1 - u - v) * v0.normal);
    jek::Vec3f tangent = jek::normalize(u * v1.tangent + v * v2.tangent + (1 - u - v) * v0.tangent);
    jek::Vec3f bitangent = jek::normalize(u * v1.bitangent + v * v2.bitangent + (1 - u - v) * v0.bitangent);
    jek::Vec2f texcoord = u * v1.texcoords + v * v2.texcoords + (1 - u - v) * v0.texcoords;

    // apply transformation to isect
    isect.normal = normalize(jek::Vec3f(transform->matrix * jek::Vec4f(normal, 0.f)));
    isect.tangent = normalize(jek::Vec3f(transform->matrix * jek::Vec4f(tangent, 0.f)));
    isect.bitangent = normalize(jek::Vec3f(transform->matrix * jek::Vec4f(bitangent, 0.f)));
    isect.texcoord = texcoord;
    isect.position = jek::Vec3f(transform->matrix * jek::Vec4f((ray_p.o + (t * ray_p.d)), 1.f));

    isect.was_found = hit;

    tmin = t;

    return hit;
}
bool dTriangle::hit(const dRay& ray) const
{
    // apply inverse transformation matrix to ray
    dRay ray_p = transform->inv_matrix * ray;

    float u, v, t;
    return intersect(ray_p, u, v, t);
}
bool dTriangle::shadow_hit(const dRay& ray, float& tmin) const
{
    float _tmin;

    // apply inverse transformation matrix to ray
    dRay ray_p = transform->inv_matrix * ray;

    float u, v, t;
    bool hit = intersect(ray_p, u, v, _tmin);

    if (_tmin < 0 || !hit) return false;

    tmin = _tmin;

    return hit;
}

Triangle::Triangle()
{
    v0.position = glm::vec3(0.f, 0.f, 0.f);
    v1.position = glm::vec3(0.f, 0.f, 1.f);
    v2.position = glm::vec3(1.f, 0.f, 0.f);

    init();
}
Triangle::Triangle(const Vertex& v0, const Vertex& v1, const Vertex& v2) :
    v0(v0), v1(v1), v2(v2)
{
    init();
}
void Triangle::init()
{
    auto v0v1 = v1.position - v0.position;
    auto v0v2 = v2.position - v0.position;
    auto ortho = cross(v0v1, v0v2);

    double area = length(ortho) * 0.5;
    inv_area = 1.0 / area;

    face_normal = normalize(ortho);
}

void intersect(const LinearBVHNode* __restrict__ nodes, const dTriangle* __restrict__ triangles, const dRay ray, Isect& isect)
{
    float		t;
    int			triangle_id;
    jek::Vec3f		normal;
    jek::Vec3f		tangent;
    jek::Vec3f		bitangent;
    float2		texcoord;
    jek::Vec3f		local_hit_point;
    float		tmin = jek::K_HUGE;
    dMaterial* material;

    jek::Vec3f invDir = jek::Vec3f(1.f / (float)ray.d.x, 1.f / (float)ray.d.y, 1.f / (float)ray.d.z);
    int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

    // Follow ray through BVH nodes to find primitive intersections //
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    Isect isect_tmp;

    while (true) {
        const LinearBVHNode* node = &nodes[currentNodeIndex];
        // Check ray against BVH node //
        if (node->bounds.hit(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node //
                for (int i = 0; i < node->nPrimitives; ++i) {

                    int id = node->primitivesOffset + i;

                    if (triangles[id].hit(ray, t, isect_tmp) && (t < tmin)) {
                        isect = isect_tmp;
                        isect.material = triangles[id].material;
                        isect.tri_id = id;
                        tmin = t;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Put far BVH node on nodesToVisit stack, advance to near node //
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    isect.t = tmin;
}
bool intersect_shadows(const LinearBVHNode* __restrict__ nodes, const dTriangle* __restrict__ triangles, const dRay ray, float& tmin)
{
    float		t;

    jek::Vec3f invDir = jek::Vec3f(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = { invDir.x < 0, invDir.y < 0, invDir.z < 0 };

    // Follow ray through BVH nodes to find primitive intersections //
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    while (true) {
        const LinearBVHNode* node = &nodes[currentNodeIndex];
        // Check ray against BVH node //
        if (node->bounds.hit(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node //
                for (int i = 0; i < node->nPrimitives; ++i) {
                    int triangle_id = node->primitivesOffset + i;
                    if (triangles[node->primitivesOffset + i].shadow_hit(ray, t) && (t < tmin)) {
                        return (true);
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            }
            else {
                // Put far BVH node on nodesToVisit stack, advance to near node //
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                }
                else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        }
        else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    return (false);
}