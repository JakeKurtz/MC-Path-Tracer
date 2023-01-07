/*
#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <glm/ext/matrix_float4x4.hpp>
#include "dRay.cuh"

struct Matrix4x4 {

    float m[4][4];

    __host__ __device__ Matrix4x4();
    __host__ __device__ Matrix4x4(float x);
    __host__ __device__ Matrix4x4(
        float t00, float t01, float t02, float t03,
        float t10, float t11, float t12, float t13,
        float t20, float t21, float t22, float t23,
        float t30, float t31, float t32, float t33);

    __host__ __device__ bool operator==(const Matrix4x4& m2) const;
    __host__ __device__ bool operator!=(const Matrix4x4& m2) const;

    __host__ __device__ Matrix4x4 operator*(const Matrix4x4 A) const;
    __host__ __device__ Matrix4x4 operator/(const Matrix4x4 A) const;
    __host__ __device__ Matrix4x4 operator+(const Matrix4x4 A) const;
    __host__ __device__ Matrix4x4 operator-(const Matrix4x4 A) const;
    __host__ __device__ Matrix4x4 operator-() const;

    __host__ __device__ Matrix4x4 transpose();

    __host__ __device__ float det() const;
    __host__ __device__ Matrix4x4 inv() const;
    __host__ __device__ void print();

    static __host__ __device__ Matrix4x4 zero(void);
    static __host__ __device__ Matrix4x4 ones(void);
    static __host__ __device__ Matrix4x4 unit(void);
};

__host__ __device__ Matrix4x4 lookAt(float3 eye, float3 look, float3 up);

__host__ __device__ Matrix4x4 perspective(float fovy, float aspect, float zNear, float zFar);

inline __host__ __device__ float4 operator*(float4 v, Matrix4x4 m)
{
    float4 out;
    out.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
    out.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
    out.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
    out.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;
    return out;
}
inline __host__ __device__ float3 operator*(float3 v, Matrix4x4 m)
{
    float4 out;
    out.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3];
    out.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3];
    out.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3];
    out.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3];
    return make_float3(out.x / out.w, out.y / out.w, out.z / out.w);
}
inline __host__ __device__ float4 operator*(Matrix4x4 m, float4 v)
{
    float4 out;
    out.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0] * v.w;
    out.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1] * v.w;
    out.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2] * v.w;
    out.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3] * v.w;
    return out;
}
inline __host__ __device__ float3 operator*(Matrix4x4 m, float3 v)
{
    float4 out;
    out.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0];
    out.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1];
    out.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2];
    out.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3];
    return make_float3(out.x / out.w, out.y / out.w, out.z / out.w);
}

inline __device__ dRay operator*(const Matrix4x4 m, const dRay r)
{
    dRay out;

    float4 o = m * make_float4(r.o.x, r.o.y, r.o.z, 1.f);
    float4 d = m * make_float4(r.d.x, r.d.y, r.d.z, 0.f);

    out.o = make_float4(o.x, o.y, o.z, 0.f);
    out.d = make_float4(d.x, d.y, d.z, 0.f);

    return out;
}
inline __device__ dRay operator*(const dRay r, const Matrix4x4 m)
{
    dRay out;

    float4 o = make_float4(r.o.x, r.o.y, r.o.z, 1.f) * m;
    float4 d = make_float4(r.d.x, r.d.y, r.d.z, 0.f) * m;

    out.o = make_float4(o.x, o.y, o.z, 0.f);
    out.d = make_float4(d.x, d.y, d.z, 0.f);

    return out;
}

static inline Matrix4x4 Matrix4x4_cast(const glm::mat4& m)
{
    return Matrix4x4(
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3],
        m[2][0], m[2][1], m[2][2], m[2][3],
        m[3][0], m[3][1], m[3][2], m[3][3]
    );
}
*/