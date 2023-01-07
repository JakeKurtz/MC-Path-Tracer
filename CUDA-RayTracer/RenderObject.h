#pragma once
#include <string>
#include "Material.h"
#include "Transform.h"
#include "Shader.h"
#include "globals.h"

class Curve;
class Mesh;

const int TYPE_CURVE = 0;
const int TYPE_TRIANGLE_MESH = 1;

class RenderObject
{
public:
	int get_id();

	virtual glm::vec3 center_of_mass() = 0;

	void set_name(std::string name);
	std::string get_name();

	void set_material(std::shared_ptr<Material> mat);
	std::shared_ptr<Material> get_material();

	void set_transform(std::shared_ptr<Transform> transform);
	std::shared_ptr<Transform> get_transform();

	int type();

	void virtual draw(Shader& shader) = 0;

protected:
	int id = gen_id();
	std::string name;
	int obj_type;

	std::shared_ptr<Material> material;
	std::shared_ptr<Transform> transform;
};

std::string construct_name_string(std::string base_name, int number);