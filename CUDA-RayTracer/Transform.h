#pragma once
#include "GLCommon.h"

#include "Matrix.h"

struct dTransform
{
	jek::Vec3f centroid;
	jek::Matrix4x4f matrix;
	jek::Matrix4x4f inv_matrix;
};

class Transform
{
public:
	Transform();
	~Transform();

	void translate(const float x, const float y, const float z);
	void translate(const glm::vec3& p);
	
	void rotate(const float angle, const glm::vec3& axis);
	void rotate(const glm::quat& q);

	void scale(const float x, const float y, const float z);
	void scale(const float s);
	void scale(const glm::vec3& s);

	void apply();
	void reset();

	dTransform* get_dptr() const;
	glm::mat4 get_matrix() const;
	glm::vec3 get_pos() const;
	glm::vec3 get_centroid() const;

	void set_matrix(const glm::mat4& matrix);
	void set_centroid(const glm::vec3& centroid);

protected:
	glm::mat4 matrix = glm::mat4(1.f);
	glm::vec3 centroid;

	glm::vec3 last_position = glm::vec3(0.f);
	glm::vec3 current_position = glm::vec3(0.f);

	glm::quat last_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);
	glm::quat current_rotation = glm::quat(1.f, 0.f, 0.f, 0.f);

	glm::vec3 s = glm::vec3(1.f);

	dTransform* dptr = nullptr;

	void update_dptr();
	void create_dptr();
	void delete_dptr();
};

