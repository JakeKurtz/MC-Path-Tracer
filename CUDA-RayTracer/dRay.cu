#include "dRay.cuh"

/*
inline __device__ dRay operator*(const jek::Matrix4x4f& m, const dRay& r)
{
    dRay out;

    jek::Vec4f o = m * jek::Vec4f(r.o.x, r.o.y, r.o.z, 1.f);
    jek::Vec4f d = m * jek::Vec4f(r.d.x, r.d.y, r.d.z, 0.f);

    out.o = jek::Vec3f(o.x, o.y, o.z);
    out.d = jek::Vec3f(d.x, d.y, d.z);

    return out;
}
inline __device__ dRay operator*(const dRay& r, const jek::Matrix4x4f& m)
{
    dRay out;

    jek::Vec4f o = jek::Vec4f(r.o.x, r.o.y, r.o.z, 1.f) * m;
    jek::Vec4f d = jek::Vec4f(r.d.x, r.d.y, r.d.z, 0.f) * m;

    out.o = jek::Vec3f(o.x, o.y, o.z);
    out.d = jek::Vec3f(d.x, d.y, d.z);

    return out;
}
*/