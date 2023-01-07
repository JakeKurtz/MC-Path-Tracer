#include "SkeletonMesh.h"

SkeletonMesh::SkeletonMesh(std::vector<glm::vec3> vertices, std::vector<unsigned int> indices)
{
	name = gen_object_name("Skeleton Mesh");
	this->vertices = vertices;
	this->indices = indices;
	this->nmb_lines = indices.size() / 2.f;

	init_buffers();
}

SkeletonMesh::SkeletonMesh(std::vector<glm::vec3> vertices, std::vector<glm::vec3> control_points, std::vector<unsigned int> indices)
{
	name = gen_object_name("Skeleton Mesh");
	this->vertices = vertices;
	this->indices = indices;
	this->control_points = control_points;
	this->nmb_lines = indices.size() / 2.f;

	init_buffers();
}

SkeletonMesh::SkeletonMesh(std::string name, std::vector<glm::vec3> vertices, std::vector<unsigned int> indices)
{
	this->name = name;
	this->vertices = vertices;
	this->indices = indices;
	this->nmb_lines = indices.size() / 2.f;

	init_buffers();
}

void SkeletonMesh::init_buffers()
{
	// ------ Control Points ------ //

// create buffers/arrays
	glGenVertexArrays(1, &VAO_cp);
	glGenBuffers(1, &VBO_cp);

	glBindVertexArray(VAO_cp);

	// load data into vertex buffers
	glBindBuffer(GL_ARRAY_BUFFER, VBO_cp);
	glBufferData(GL_ARRAY_BUFFER, control_points.size() * sizeof(glm::vec3), &control_points[0], GL_STATIC_DRAW);

	// set the vertex attribute pointers
	// vertex Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

	glBindVertexArray(0);

	// create buffers/arrays
	glGenVertexArrays(1, &VAO_curve);
	glGenBuffers(1, &VBO_curve);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO_curve);

	// load data into vertex buffers
	if (vertices.empty()) {

	}
	else {
		glBindBuffer(GL_ARRAY_BUFFER, VBO_curve);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
	}

	if (indices.empty()) {

	}
	else {
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
	}

	// set the vertex attribute pointers
	// vertex Positions
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
	/*
	// vertex normals
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	// vertex texture coords
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texCoords));
	// vertex tangent
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, tangent));
	// vertex bitangent
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, bitangent));
	*/
	glBindVertexArray(0);
}

void SkeletonMesh::draw(Shader& shader)
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	// draw mesh
	//glLineWidth(4.f);
	//glBindVertexArray(VAO_curve);
	//glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
	//glBindVertexArray(0);

	// draw mesh
	//glLineWidth(4.f);
	//glBindVertexArray(VAO_curve);
	//glDrawElements(GL_LINES, indices.size(), GL_UNSIGNED_INT, 0);
	//glBindVertexArray(0);

	// draw control points
	glPointSize(6);
	glBindVertexArray(VAO_cp);
	glDrawArrays(GL_POINTS, 0, control_points.size());
	glBindVertexArray(0);
}

void SkeletonMesh::update()
{
	init_buffers();
}
