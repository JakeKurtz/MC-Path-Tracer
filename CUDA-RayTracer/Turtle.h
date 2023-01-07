#pragma once

#include "GLCommon.h"
#include "Curve.h"
#include "SkeletonMesh.h"
#include "LModule.h"
#include <cmath>
#include <stack>
#include "Model.h"

struct TurtleVertex {

	TurtleVertex() {

	}

	TurtleVertex(const TurtleVertex& v) {
		dir = v.dir;
		right = v.right;
		up = v.up;
		pos = v.pos;
		step_size = v.step_size;
		thickness = v.thickness;
		angle = v.angle;
		vertex_index = v.vertex_index;
	}

	glm::vec3 dir;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 pos;
	float step_size;
	float thickness;
	float angle;
	int vertex_index;
};

struct TurtleState {
	glm::vec3 dir;
	glm::vec3 right;
	glm::vec3 up;
	glm::vec3 pos;
	float step_size;
	float thickness;
	float angle;
	int vertex_index;
};

class Turtle
{
public:
	Turtle(glm::vec3 pos = glm::vec3(0,0,0), glm::vec3 dir = glm::vec3(0,1,0));
	
	void set_command_str(std::string cmd_str);

	std::shared_ptr<Curve> get_curve();
	std::shared_ptr<Mesh> get_mesh();

//private:

	void set_default_state();

	bool build_curve();
	bool execute_op(const char symbol, const std::vector<float> parameters);

	void F(float l);
	void f(float l);

	void apply_yaw(float a);
	void apply_pitch(float a);
	void apply_roll(float a);

	void push_state();
	void pop_state();

	void update_orientation(glm::mat4 mat);
	void update_vectors(glm::mat4 mat);

	void gen_tube(int granularity, TurtleVertex start, TurtleVertex end);

	void gen_hemisphere(glm::vec3 orientation, std::vector<Vertex> disk_verts, TurtleVertex v);
	
	Vertex get_midway_point(Vertex v0, Vertex v1);
	std::vector<std::shared_ptr<Triangle>> midpoint_subdivide(std::shared_ptr<Triangle> t);

	std::vector<Vertex> gen_unit_disk(int granularity, TurtleVertex pos);

	void bloat_vert(Vertex& v, const TurtleVertex& tv, const glm::vec3 orientation);

	std::shared_ptr<Curve> curve;
	std::shared_ptr<Mesh> mesh;
	std::shared_ptr<Material> mat = nullptr;

	std::vector<std::shared_ptr<Triangle>> triangles;

	std::vector<TurtleVertex> vertices;
	std::vector<Vertex> vertices_triangles;
	std::vector<glm::vec3> control_points;
	std::vector<unsigned int> indices;
	std::vector<unsigned int> indices_triangles;

	// Attributes
	glm::vec3 pos = glm::vec3(0.f);
	glm::vec3 dir = glm::vec3(0.f,1.f,0.f);

	glm::mat4 look_at = glm::mat4(1.f);
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 world_up = glm::vec3(0.f, 0.f, 1.f);

	std::stack<TurtleVertex> state;
	TurtleVertex curr_state;

	std::string cmd_str = "";

	float default_thickness_scale = 0.5;

	float default_step_size = 1.f;
	float default_angle = 28.f;
	float default_step_size_scale = 0.5;
	float default_thickness = 0.1;
};

