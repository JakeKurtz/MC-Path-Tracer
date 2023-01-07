#include "Transform.h"
#include "CudaHelpers.h"

Transform::Transform() 
{
	centroid = glm::vec3(0.f);
	current_position = glm::vec3(0.f);
	s = glm::vec3(1.f);
	matrix = glm::mat4(1.f);
	create_dptr();
}
Transform::~Transform()
{
//	delete_dptr();
}

void Transform::translate(const float x, const float y, const float z)
{
	(this)->current_position = glm::vec3(x,y,z);
}
void Transform::translate(const glm::vec3& p)
{
	(this)->current_position = p;
}

void Transform::rotate(const float angle, const glm::vec3& axis)
{
	(this)->current_rotation = glm::angleAxis(angle, axis);
}
void Transform::rotate(const glm::quat& q)
{
	(this)->current_rotation = q;
}

void Transform::scale(const float x, const float y, const float z)
{
	(this)->s = glm::vec3(x, y, z);
}
void Transform::scale(const float s)
{
	(this)->s = glm::vec3(s);
}
void Transform::scale(const glm::vec3& s)
{
	(this)->s = s;
}

void Transform::apply()
{
	matrix = glm::translate(glm::mat4(1.f), centroid);

	// Translate
	last_position = current_position + last_position;
	current_position = glm::vec3(0.f);
	matrix = glm::translate(matrix, last_position);

	// Rotate
	last_rotation = current_rotation * last_rotation;
	current_rotation = glm::mat4(1.f);
	matrix = matrix * glm::mat4_cast(last_rotation);

	// Scale
	matrix = glm::scale(matrix, s);
	matrix = glm::translate(matrix, -centroid);

	update_dptr();
}
void Transform::reset()
{
	centroid = glm::vec3(0.f);

	last_position = glm::vec3(0.f);
	current_position = glm::vec3(0.f);

	s = glm::vec3(1.f);

	matrix = glm::mat4(1.f);

	last_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
	current_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);

	update_dptr();
}

dTransform* Transform::get_dptr() const
{
	return dptr;
}
glm::mat4 Transform::get_matrix() const
{
	return matrix;
}
glm::vec3 Transform::get_pos() const
{
	return last_position;
}
glm::vec3 Transform::get_centroid() const
{
	return centroid;
}

void Transform::set_matrix(const glm::mat4& matrix)
{
	(this)->matrix = matrix;
	update_dptr();
}
void Transform::set_centroid(const glm::vec3& centroid)
{
	(this)->centroid = centroid;
	update_dptr();
}

void Transform::update_dptr()
{
	jek::Matrix4x4f mat = jek::Matrix4x4f(matrix);
	(this)->dptr->matrix = mat;
	(this)->dptr->inv_matrix = jek::inv(mat);
	(this)->dptr->centroid = jek::Vec3f(centroid);
}
void Transform::create_dptr()
{
	checkCudaErrors(cudaMallocManaged(&dptr, sizeof(dTransform)));
	jek::Matrix4x4f mat = jek::Matrix4x4f(matrix);
	update_dptr();
}
void Transform::delete_dptr()
{
	if (dptr != nullptr) {
		checkCudaErrors(cudaFree(dptr));
		dptr = nullptr;
	}
}