#include "Quaternion.h"
/*
template<typename T>
bool Quaternion<T>::operator==(const Quaternion& q) const
{
	return (x==q.x && y == q.y && z == q.z && w == q.w);
}

template<typename T>
bool Quaternion<T>::operator!=(const Quaternion& q) const
{
	return (x != q.x || y != q.y || z != q.z || w != q.w);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion q) const
{
	T x_p = x*q.x - y*q.y - z*q.z - w*q.w;
	T y_p = x*q.y + y*q.x + z*q.w - w*q.z;
	T z_p = x*q.z - y*q.w + z*q.x + w*q.y;
	T w_p = x*q.w + y*q.z - z*q.y + w*q.x;

	return Quaternion<T>(x_p, y_p, z_p, w_p);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(const T s) const
{
	return Quaternion<T>(s*x, s*y, s*z, s*w);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator/(const T s) const
{
	return Quaternion<T>(s/x, s/y, s/z, s/w);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator+(const Quaternion q) const
{
	return Quaternion<T>(x + q.x, y + q.y, z + q.z, w + q.w);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator-(const Quaternion q) const
{
	return Quaternion<T>(x - q.x, y - q.y, z - q.z, w - q.w);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator-() const
{
	return Quaternion<T>(-x, -y, -z, -w);
}

template<typename T>
Quaternion<T> Quaternion<T>::normalize() const
{
	auto norm = glm::sqrt(x*x + y*y + z*z + w*w);
	auto q = *this;
	return q/norm;
}

template<typename T>
Quaternion<T> Quaternion<T>::conjugate() const
{
	return Quaternion<T>(x, -y, -z, -w);
}


template<typename T>
void Quaternion<T>::angle_axis(T& angle, glm::vec3& axis)
{
}

template<typename T>
void Quaternion<T>::to_mat4(glm::mat4& mat)
{
}

template<typename T>
void Quaternion<T>::print()
{
	cout << "Quaternion(" << x << ", " << y << ", " << z << ", " << w << ")" << endl;
}

template<typename T>
Quaternion<T> Quaternion<T>::unit(void)
{
	return Quaternion();
}

template<typename T>
Quaternion<T> Quaternion<T>::normalize(const Quaternion q)
{
	return Quaternion();
}
*/