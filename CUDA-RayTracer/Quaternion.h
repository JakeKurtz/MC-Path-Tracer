#pragma once

#include "GLCommon.h"

template <typename T> struct Quaternion
{
	Quaternion() {
		x = 1;
		y = 0;
		z = 0;
		w = 0;
	}

	Quaternion(T x, T y, T z, T w) :
		x(x), y(y), z(z), w(w) {}

	Quaternion(glm::vec3 v, T w) :
		x(v.x), y(v.y), z(v.z), w(w) {}

	Quaternion(const Quaternion<T>& q) :
		x(q.x), y(q.y), z(q.z), w(q.w) {}

	T x;
	T y; 
	T z;
	T w;

	bool operator==(const Quaternion& q) const;
	bool operator!=(const Quaternion& q) const;

	Quaternion<T> operator*(const Quaternion q) const;
	Quaternion<T> operator*(const T s) const;

	Quaternion<T> operator/(const T s) const;

	Quaternion<T> operator+(const Quaternion q) const;
	Quaternion<T> operator-(const Quaternion q) const;

	Quaternion<T> operator-() const;

	void normalize();
	Quaternion<T> conjugate() const;

	void angle_axis(T &angle, glm::vec<3, T>&axis);
	void to_mat4(glm::mat4 &mat);

	void print();

	static Quaternion<T> unit(void);
	static Quaternion<T> normalize(const Quaternion q);
};

template<typename T>
inline bool Quaternion<T>::operator==(const Quaternion& q) const
{
	return (x == q.x && y == q.y && z == q.z && w == q.w);
}

template<typename T>
inline bool Quaternion<T>::operator!=(const Quaternion& q) const
{
	return (x != q.x || y != q.y || z != q.z || w != q.w);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator*(const Quaternion q) const
{
	T w_p = w*q.w - x*q.x - y*q.y - z*q.z;
	T x_p = w*q.x + x*q.w + y*q.z - z*q.y;
	T y_p = w*q.y - x*q.z + y*q.w + z*q.x;
	T z_p = w*q.z + x*q.y - y*q.x + z*q.w;

	return Quaternion<T>(x_p, y_p, z_p, w_p);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator*(const T s) const
{
	return Quaternion<T>(s * x, s * y, s * z, s * w);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator/(const T s) const
{
	return Quaternion<T>(s / x, s / y, s / z, s / w);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator+(const Quaternion q) const
{
	return Quaternion<T>(x + q.x, y + q.y, z + q.z, w + q.w);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator-(const Quaternion q) const
{
	return Quaternion<T>(x - q.x, y - q.y, z - q.z, w - q.w);
}

template<typename T>
inline Quaternion<T> Quaternion<T>::operator-() const
{
	return Quaternion<T>(-x, -y, -z, -w);
}

template<typename T>
inline void Quaternion<T>::normalize()
{
	T norm = glm::sqrt(x * x + y * y + z * z + w * w);
	x = x / norm;
	y = y / norm;
	z = z / norm;
	w = w / norm;
}

template<typename T>
Quaternion<T> Quaternion<T>::conjugate() const
{
	return Quaternion<T>(x, -y, -z, -w);
}


template<typename T>
inline void Quaternion<T>::angle_axis(T& angle, glm::vec<3, T>& axis)
{
	if (glm::vec<3, T>(x, y, z) == glm::vec<3, T>(0.f)) {
		axis = glm::vec<3, T>(1,0,0);
	}
	else {
		axis = glm::normalize(glm::vec<3, T>(x, y, z));
	}
	angle = glm::radians(2.f * atan2f(glm::sqrt(x*x + y*y + z*z), w));
}

template<typename T>
inline void Quaternion<T>::to_mat4(glm::mat4& mat)
{
	auto norm = glm::sqrt(x*x + y*y + z*z + w*w);
	auto s = 2.f / (norm*norm);
	/*
	auto m00 = 1.f - s*(z*z + w*w);
	auto m01 = s*(y*z - w*x);
	auto m02 = s*(y*w + z*x);
	auto m10 = s*(y*z + w*x);
	auto m11 = 1.f - s*(y*y + w*w);
	auto m12 = s*(z*w - y*x);
	auto m20 = s*(y*w - z*x);
	auto m21 = s*(z*w + y*x);
	auto m22 = 1.f - s*(y*y + z*z);
	*/

	auto m00 = 1.f - s*(y*y + z*z);
	auto m01 = s*(x*y - z*w);
	auto m02 = s*(x*z + y*w);
	auto m10 = s*(x*y + z*w);
	auto m11 = 1.f - s*(x*x + z*z);
	auto m12 = s*(y*z - x*w);
	auto m20 = s*(x*z - y*w);
	auto m21 = s*(y*z + x*w);
	auto m22 = 1.f - s*(x*x + y*y);

	mat = glm::mat4(
		m00, m10, m20, 0.f,
		m01, m11, m21, 0.f, 
		m02, m12, m22, 0.f, 
		0.f, 0.f, 0.f, 1.f
	);
}

template<typename T>
inline void Quaternion<T>::print()
{
	std::cout << "Quaternion(" << x << ", " << y << ", " << z << ", " << w << ")" << std::endl;
}

template<typename T>
inline Quaternion<T> Quaternion<T>::unit(void)
{
	return Quaternion();
}

template<typename T>
inline Quaternion<T> Quaternion<T>::normalize(const Quaternion q)
{
	return Quaternion();
}