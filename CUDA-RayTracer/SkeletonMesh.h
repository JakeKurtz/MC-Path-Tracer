#pragma once
#include "GLCommon.h"
#include "globals.h"
#include "Curve.h"
#include "Mesh.h"

class SkeletonMesh : public Curve
{
public:
	SkeletonMesh(std::vector<glm::vec3> vertices, std::vector<unsigned int> indices);
	SkeletonMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> control_points, std::vector<unsigned int> indices);
	SkeletonMesh(std::string name, std::vector<glm::vec3> vertices, std::vector<unsigned int> indices);

	void init_buffers();
	void draw(Shader& shader);

	void update();
	
	//std::vector<glm::vec3> vertices;
	//std::vector<unsigned int> indices;

	int nmb_lines;
	
	unsigned int VAO, VBO, EBO;
};

