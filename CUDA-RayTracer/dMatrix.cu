/*
#include "dMatrix.cuh"
#include "dMath.cuh"

__host__ __device__ Matrix4x4::Matrix4x4()
{
    m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1.f;

    m[0][1] = m[0][2] = m[0][3] = m[1][0] =
        m[1][2] = m[1][3] = m[2][0] = m[2][1] =
        m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0.f;
}

__host__ __device__ Matrix4x4::Matrix4x4(float x)
{
    m[0][0] = m[1][1] = m[2][2] = m[3][3] =
        m[0][1] = m[0][2] = m[0][3] = m[1][0] =
        m[1][2] = m[1][3] = m[2][0] = m[2][1] =
        m[2][3] = m[3][0] = m[3][1] = m[3][2] = x;
}

__host__ __device__ Matrix4x4::Matrix4x4(
    float t00, float t01, float t02, float t03,
    float t10, float t11, float t12, float t13,
    float t20, float t21, float t22, float t23,
    float t30, float t31, float t32, float t33)
{
    m[0][0] = t00; m[0][1] = t01; m[0][2] = t02; m[0][3] = t03;
    m[1][0] = t10; m[1][1] = t11; m[1][2] = t12; m[1][3] = t13;
    m[2][0] = t20; m[2][1] = t21; m[2][2] = t22; m[2][3] = t23;
    m[3][0] = t30; m[3][1] = t31; m[3][2] = t32; m[3][3] = t33;
}

__host__ __device__ bool Matrix4x4::operator==(const Matrix4x4& m2) const {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (m[i][j] != m2.m[i][j]) return false;
    return true;
}
__host__ __device__ bool Matrix4x4::operator!=(const Matrix4x4& m2) const {
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (m[i][j] != m2.m[i][j]) return true;
    return false;
}

__host__ __device__ Matrix4x4 Matrix4x4::operator*(const Matrix4x4 A) const
{
    Matrix4x4 out;
    for (int i = 0; i < 4; i++) {
        const float ai0 = m[i][0], ai1 = m[i][1], ai2 = m[i][2], ai3 = m[i][3];
        out.m[i][0] = ai0 * A.m[0][0] + ai1 * A.m[1][0] + ai2 * A.m[2][0] + ai3 * A.m[3][0];
        out.m[i][1] = ai0 * A.m[0][1] + ai1 * A.m[1][1] + ai2 * A.m[2][1] + ai3 * A.m[3][1];
        out.m[i][2] = ai0 * A.m[0][2] + ai1 * A.m[1][2] + ai2 * A.m[2][2] + ai3 * A.m[3][2];
        out.m[i][3] = ai0 * A.m[0][3] + ai1 * A.m[1][3] + ai2 * A.m[2][3] + ai3 * A.m[3][3];
    }
    return out;
}
__host__ __device__ Matrix4x4 Matrix4x4::operator/(const Matrix4x4 A) const
{
    return Matrix4x4(0.f);
}
__host__ __device__ Matrix4x4 Matrix4x4::operator+(const Matrix4x4 A) const
{
    return Matrix4x4(0.f);
}
__host__ __device__ Matrix4x4 Matrix4x4::operator-(const Matrix4x4 A) const
{
    return Matrix4x4(0.f);
}
__host__ __device__ Matrix4x4 Matrix4x4::operator-() const
{
    return Matrix4x4(0.f);
}

__host__ __device__ Matrix4x4 Matrix4x4::transpose() {
    return Matrix4x4(
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]
    );
}

__host__ __device__ float Matrix4x4::det() const
{
    return 0.f;
}
__host__ __device__ Matrix4x4 Matrix4x4::inv() const
{
    Matrix4x4 inv;
    double det;

    inv.m[0][0] = m[1][1] * m[2][2] * m[3][3] -
        m[1][1] * m[2][3] * m[3][2] -
        m[2][1] * m[1][2] * m[3][3] +
        m[2][1] * m[1][3] * m[3][2] +
        m[3][1] * m[1][2] * m[2][3] -
        m[3][1] * m[1][3] * m[2][2];

    inv.m[1][0] = -m[1][0] * m[2][2] * m[3][3] +
        m[1][0] * m[2][3] * m[3][2] +
        m[2][0] * m[1][2] * m[3][3] -
        m[2][0] * m[1][3] * m[3][2] -
        m[3][0] * m[1][2] * m[2][3] +
        m[3][0] * m[1][3] * m[2][2];

    inv.m[2][0] = m[1][0] * m[2][1] * m[3][3] -
        m[1][0] * m[2][3] * m[3][1] -
        m[2][0] * m[1][1] * m[3][3] +
        m[2][0] * m[1][3] * m[3][1] +
        m[3][0] * m[1][1] * m[2][3] -
        m[3][0] * m[1][3] * m[2][1];

    inv.m[3][0] = -m[1][0] * m[2][1] * m[3][2] +
        m[1][0] * m[2][2] * m[3][1] +
        m[2][0] * m[1][1] * m[3][2] -
        m[2][0] * m[1][2] * m[3][1] -
        m[3][0] * m[1][1] * m[2][2] +
        m[3][0] * m[1][2] * m[2][1];

    inv.m[0][1] = -m[0][1] * m[2][2] * m[3][3] +
        m[0][1] * m[2][3] * m[3][2] +
        m[2][1] * m[0][2] * m[3][3] -
        m[2][1] * m[0][3] * m[3][2] -
        m[3][1] * m[0][2] * m[2][3] +
        m[3][1] * m[0][3] * m[2][2];

    inv.m[1][1] = m[0][0] * m[2][2] * m[3][3] -
        m[0][0] * m[2][3] * m[3][2] -
        m[2][0] * m[0][2] * m[3][3] +
        m[2][0] * m[0][3] * m[3][2] +
        m[3][0] * m[0][2] * m[2][3] -
        m[3][0] * m[0][3] * m[2][2];

    inv.m[2][1] = -m[0][0] * m[2][1] * m[3][3] +
        m[0][0] * m[2][3] * m[3][1] +
        m[2][0] * m[0][1] * m[3][3] -
        m[2][0] * m[0][3] * m[3][1] -
        m[3][0] * m[0][1] * m[2][3] +
        m[3][0] * m[0][3] * m[2][1];

    inv.m[3][1] = m[0][0] * m[2][1] * m[3][2] -
        m[0][0] * m[2][2] * m[3][1] -
        m[2][0] * m[0][1] * m[3][2] +
        m[2][0] * m[0][2] * m[3][1] +
        m[3][0] * m[0][1] * m[2][2] -
        m[3][0] * m[0][2] * m[2][1];

    inv.m[0][2] = m[0][1] * m[1][2] * m[3][3] -
        m[0][1] * m[1][3] * m[3][2] -
        m[1][1] * m[0][2] * m[3][3] +
        m[1][1] * m[0][3] * m[3][2] +
        m[3][1] * m[0][2] * m[1][3] -
        m[3][1] * m[0][3] * m[1][2];

    inv.m[1][2] = -m[0][0] * m[1][2] * m[3][3] +
        m[0][0] * m[1][3] * m[3][2] +
        m[1][0] * m[0][2] * m[3][3] -
        m[1][0] * m[0][3] * m[3][2] -
        m[3][0] * m[0][2] * m[1][3] +
        m[3][0] * m[0][3] * m[1][2];

    inv.m[2][2] = m[0][0] * m[1][1] * m[3][3] -
        m[0][0] * m[1][3] * m[3][1] -
        m[1][0] * m[0][1] * m[3][3] +
        m[1][0] * m[0][3] * m[3][1] +
        m[3][0] * m[0][1] * m[1][3] -
        m[3][0] * m[0][3] * m[1][1];

    inv.m[3][2] = -m[0][0] * m[1][1] * m[3][2] +
        m[0][0] * m[1][2] * m[3][1] +
        m[1][0] * m[0][1] * m[3][2] -
        m[1][0] * m[0][2] * m[3][1] -
        m[3][0] * m[0][1] * m[1][2] +
        m[3][0] * m[0][2] * m[1][1];

    inv.m[0][3] = -m[0][1] * m[1][2] * m[2][3] +
        m[0][1] * m[1][3] * m[2][2] +
        m[1][1] * m[0][2] * m[2][3] -
        m[1][1] * m[0][3] * m[2][2] -
        m[2][1] * m[0][2] * m[1][3] +
        m[2][1] * m[0][3] * m[1][2];

    inv.m[1][3] = m[0][0] * m[1][2] * m[2][3] -
        m[0][0] * m[1][3] * m[2][2] -
        m[1][0] * m[0][2] * m[2][3] +
        m[1][0] * m[0][3] * m[2][2] +
        m[2][0] * m[0][2] * m[1][3] -
        m[2][0] * m[0][3] * m[1][2];

    inv.m[2][3] = -m[0][0] * m[1][1] * m[2][3] +
        m[0][0] * m[1][3] * m[2][1] +
        m[1][0] * m[0][1] * m[2][3] -
        m[1][0] * m[0][3] * m[2][1] -
        m[2][0] * m[0][1] * m[1][3] +
        m[2][0] * m[0][3] * m[1][1];

    inv.m[3][3] = m[0][0] * m[1][1] * m[2][2] -
        m[0][0] * m[1][2] * m[2][1] -
        m[1][0] * m[0][1] * m[2][2] +
        m[1][0] * m[0][2] * m[2][1] +
        m[2][0] * m[0][1] * m[1][2] -
        m[2][0] * m[0][2] * m[1][1];

    det = m[0][0] * inv.m[0][0] + m[0][1] * inv.m[1][0] + m[0][2] * inv.m[2][0] + m[0][3] * inv.m[3][0];

    if (det == 0)
        return Matrix4x4(0.f);

    det = 1.0 / det;

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            inv.m[i][j] *= det;

    return inv;
}

__host__ __device__ void Matrix4x4::print()
{
    printf("[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n\n", 
        m[0][0], m[0][1], m[0][2], m[0][3],
        m[1][0], m[1][1], m[1][2], m[1][3], 
        m[2][0], m[2][1], m[2][2], m[2][3], 
        m[3][0], m[3][1], m[3][2], m[3][3]
    );
}

__host__ __device__ Matrix4x4 Matrix4x4::zero(void)
{
    return Matrix4x4(0.f);
}
__host__ __device__ Matrix4x4 Matrix4x4::ones(void)
{
    return Matrix4x4(1.f);
}
__host__ __device__ Matrix4x4 Matrix4x4::unit(void)
{
    return Matrix4x4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    );
}

__host__ __device__ Matrix4x4 lookAt(float3 eye, float3 look, float3 up)
{
    float3 d = normalize(look - eye);
    float3 r = normalize(cross(d, up));
    float3 u = cross(r, d);

    Matrix4x4 Result;

    Result.m[0][0] = r.x;
    Result.m[1][0] = r.y;
    Result.m[2][0] = r.z;
    Result.m[0][1] = u.x;
    Result.m[1][1] = u.y;
    Result.m[2][1] = u.z;
    Result.m[0][2] = -d.x;
    Result.m[1][2] = -d.y;
    Result.m[2][2] = -d.z;
    Result.m[3][0] = -dot(r, eye);
    Result.m[3][1] = -dot(u, eye);
    Result.m[3][2] = dot(d, eye);

    return Result;
}
__host__ __device__ Matrix4x4 perspective(float fovy, float aspect, float zNear, float zFar)
{
    float const tanHalfFovy = tan(fovy / 2.f);

    Matrix4x4 Result;
    Result.m[0][0] = 1.f / (aspect * tanHalfFovy);
    Result.m[1][1] = 1.f / (tanHalfFovy);
    Result.m[2][2] = -(zFar + zNear) / (zFar - zNear);
    Result.m[2][3] = -1.f;
    Result.m[3][2] = -(2.f * zFar * zNear) / (zFar - zNear);
    return Result;
}
*/