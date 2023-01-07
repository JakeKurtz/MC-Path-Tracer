#pragma once
#include "Camera.h"

class PerspectiveCamera : public Camera
{
public:
	PerspectiveCamera();
	PerspectiveCamera(float yfov, float znear, float zfar);
	PerspectiveCamera(glm::vec3 position, float yfov, float znear, float zfar);
private:
	void update_proj_mat();
};

