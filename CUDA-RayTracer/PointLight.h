#pragma once

#include "Light.h"

class PointLight : public Light
{
private:
	glm::vec3 position;

public:
	PointLight();
	PointLight(glm::vec3 position, glm::vec3 color);

	glm::vec3 getPosition();
	void setPosition(glm::vec3 _position);
};

