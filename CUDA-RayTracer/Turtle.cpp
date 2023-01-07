#include "Turtle.h"

Turtle::Turtle(glm::vec3 pos, glm::vec3 dir)
{
	(this)->pos = pos;
	(this)->dir = dir;

	set_default_state();
}

void Turtle::set_command_str(std::string cmd_str)
{
	(this)->cmd_str = cmd_str;
}

std::shared_ptr<Curve> Turtle::get_curve()
{
	return curve;
}

std::shared_ptr<Mesh> Turtle::get_mesh()
{
	return mesh;
}

void get_symbol_range(const boost::string_ref s, int& start_pos, int& end_pos)
{
	end_pos = start_pos + 1;

	if (end_pos < s.size()) {
		if (s[start_pos + 1] == '(') {
			char c = ' ';
			while (c != ')') {
				c = s[end_pos];
				end_pos++;
			}
		}
	}
}

void load_parameters(const boost::string_ref input_string, std::vector<float>& parameters)
{
	if (input_string.size() > 0) {
		std::string parameters_raw = trim_brackets(input_string);
		std::stringstream ss(parameters_raw);
		while (ss.good())
		{
			std::string p_str;
			getline(ss, p_str, ',');

			try {
				parameters.push_back(std::stof(p_str));
			}
			catch (const std::exception& e) {
				std::cerr << "Turtle Error: the parameter \"" << p_str << "\" is an invalid number. Value is set to 0.0" << std::endl;
				parameters.push_back(0.f);
			}
		}
	}
}

void Turtle::set_default_state()
{
	curr_state.pos = pos;
	curr_state.dir = dir;

	curr_state.angle = default_angle;
	curr_state.step_size = default_step_size;
	curr_state.thickness = default_thickness;

	curr_state.vertex_index = 0;

	update_orientation(glm::mat4(1.f));
}

bool Turtle::build_curve()
{
	bool build_success = true;

	set_default_state();

	//vertices.clear();
	//indices.clear();
	//control_points.clear();

	vertices_triangles.clear();
	indices_triangles.clear();
	triangles.clear();

	if (cmd_str == "") {
		std::cerr << "Turtle Error: the command string is empty";
		build_success =  false;
	}
	else {

		TurtleVertex tv = curr_state;
		vertices.push_back(tv);

		std::string cmd_str_ref = cmd_str;

		char symbol = '\0';

		int i = 0;
		while (i < cmd_str_ref.size())
		{
			symbol = cmd_str_ref.at(i);

			int sym_start = i, sym_end = i;
			get_symbol_range(cmd_str_ref, sym_start, sym_end);

			int sym_length = sym_end - sym_start;

			std::vector<float> parameters;
			load_parameters(cmd_str_ref.substr(sym_start + 1, sym_length - 1), parameters);

			execute_op(symbol, parameters);

			i += sym_length;
		}

		if (!state.empty()) {
			std::cerr << "Turtle Warning: states remain on the stack! Ensure the brackets \'[\' and \']\' are balanced in the replacement string." << std::endl;
		}

		if (curve == nullptr) {
			//curve = std::make_shared<SkeletonMesh>(vertices, control_points, indices);
			//curve = std::make_shared<SkeletonMesh>();// (vertices_triangles, indices_triangles);
		}
		else {
			//curve->set_vertices(vertices);
			//curve->set_indices(indices);
			//curve->update();
		}

		if (mesh == nullptr) {
			mat = std::make_shared<Material>();
			mesh = std::make_shared<Mesh>(triangles, mat);
		}
		else {
			//mesh->set_triangles(triangles);
			//mesh->center_of_mass();
		}
	}
	return build_success;
}

bool Turtle::execute_op(const char symbol, const std::vector<float> p)
{
	bool success = true;
	switch (symbol) {
	case 'F':
		if (p.size() >= 1) {
			float l = p[0];
			F(l);
		}
		else {
			F(curr_state.step_size);
		}
		break;
	case 'H':
		break;
	case 'G':
		break;
	case 'f':
		if (p.size() >= 1) {
			float l = p[0];
			f(l);
		}
		else {
			f(curr_state.step_size);
		}
		break;
	case 'h':
		break;
	case 'T':
		break;
	case '+':
		if (p.size() >= 1) {
			apply_yaw(p[0]);
		}
		else {
			apply_yaw(curr_state.angle);
		}
		break;
	case '-':
		if (p.size() >= 1) {
			apply_yaw(-p[0]);
		}
		else {
			apply_yaw(-curr_state.angle);
		}
		break;
	case '&':
		if (p.size() >= 1) {
			apply_pitch(p[0]);
		}
		else {
			apply_pitch(curr_state.angle);
		}
		break;
	case '^':
		if (p.size() >= 1) {
			apply_pitch(-p[0]);
		}
		else {
			apply_pitch(-curr_state.angle);
		}
		break;
	case '\\':
		if (p.size() >= 1) {
			apply_roll(p[0]);
		}
		else {
			apply_roll(curr_state.angle);
		}
		break;
	case '/':
		if (p.size() >= 1) {
			apply_roll(-p[0]);
		}
		else {
			apply_roll(-curr_state.angle);
		}
		break;
	case '|':
		apply_yaw(180);
		break;
	case '*':
		apply_roll(180);
		break;
	case '~':
		break;
	case '\"':
		if (p.size() >= 1) {
			curr_state.step_size *= p[0];
		}
		else {
			curr_state.step_size *= default_step_size_scale;
		}
		break;
	case '!':
		if (p.size() >= 1) {
			curr_state.thickness *= p[0];
		}
		else {
			curr_state.thickness *= default_thickness_scale;
		}
		break;
	case ';':
		break;
	case '_':
		if (p.size() >= 1) {
			curr_state.step_size / p[0];
		}
		else {
			curr_state.step_size / default_step_size_scale;
		}
		break;
	case '?':
		break;
	case '@':
		break;
	case '\'':
		break;
	case '#':
		break;
	case '%':
		break;
	case '$':
		update_orientation(glm::mat4(1.f));
		break;
	case '[':
		push_state();
		break;
	case ']':
		pop_state();
		break;
	case '{':
		break;
	case '.':
		break;
	case '}':
		break;
	default:
		success = false;
	}
	return success;
}

void Turtle::F(float l)
{
	auto vertex_index = curr_state.vertex_index;

	// Translate turtle
	curr_state.pos = l * curr_state.dir + curr_state.pos;

	TurtleVertex tv = curr_state;

	// Push new vertex position to the vertex list
	vertices.push_back(tv);
	curr_state.vertex_index = vertices.size()-1;

	// New line
	int start = vertex_index;
	int end = curr_state.vertex_index;

	//indices.push_back(start);
	//indices.push_back(end);

	gen_tube(4, vertices[start], vertices[end]);
}

void Turtle::f(float l)
{
	// Translate turtle
	curr_state.pos = l * curr_state.dir + curr_state.pos;

	TurtleVertex tv = curr_state;

	// Push new vertex position to the vertex list
	vertices.push_back(tv);
	curr_state.vertex_index = vertices.size() - 1;
}

void Turtle::apply_yaw(float a)
{
	update_vectors(glm::rotate(glm::mat4(1.f), glm::radians(a), curr_state.up));
}

void Turtle::apply_pitch(float a)
{
	update_vectors(glm::rotate(glm::mat4(1.f), glm::radians(a), curr_state.right));
}

void Turtle::apply_roll(float a)
{
	update_vectors(glm::rotate(glm::mat4(1.f), glm::radians(a), curr_state.dir));
}

void Turtle::push_state()
{
	state.push(curr_state);
}

void Turtle::pop_state()
{
	if (!state.empty()) {
		curr_state = state.top();
		state.pop();
	}
	else {
		std::cerr << "Turtle Error: tried to pop an empty stack! Ensure the brackets \'[\' and \']\' are balanced in the replacement string." << std::endl;
	}
}

void Turtle::update_vectors(glm::mat4 mat)
{
	curr_state.dir = glm::normalize(mat * glm::vec4(curr_state.dir, 0.f));
	curr_state.right = glm::normalize(mat * glm::vec4(curr_state.right, 0.f));
	curr_state.up = glm::normalize(mat * glm::vec4(curr_state.up, 0.f));
}

std::vector<Vertex> Turtle::gen_unit_disk(int granularity, TurtleVertex v)
{
	glm::mat4 matrix = glm::mat4(glm::vec4(curr_state.right, 0), glm::vec4(curr_state.dir, 0), glm::vec4(curr_state.up, 0), glm::vec4(v.pos, 1));
	std::vector<Vertex> out;
	float angle = 360.f / granularity;
	for (float theta = 0.f; glm::round(theta) < 360.f; theta += angle)
	{
		float r_theta = glm::radians(theta);

		float x = v.thickness * std::cos(r_theta);
		float y = 0.f;
		float z = v.thickness * std::sin(r_theta);

		Vertex v;

		v.position = glm::vec3(matrix * glm::vec4(x, y, z, 1.f));
		v.normal = glm::normalize(glm::vec3(matrix * glm::vec4(glm::vec3(x, y, z), 0.f)));

		out.push_back(v);
	}
	return out;
}

void Turtle::gen_hemisphere(glm::vec3 orientation, std::vector<Vertex> disk_verts, TurtleVertex v)
{
	glm::mat4 matrix = glm::mat4(glm::vec4(curr_state.right, 0), glm::vec4(curr_state.dir, 0), glm::vec4(curr_state.up, 0), glm::vec4(v.pos, 1));

	Vertex v0;
	v0.position = matrix * glm::vec4(orientation * v.thickness, 1.f);
	v0.normal = glm::normalize(matrix * glm::vec4(orientation, 0.f));

	std::vector<std::shared_ptr<Triangle>> hemi_tmp;
	std::vector<std::shared_ptr<Triangle>> hemi_subdiv;

	// basis
	for (int i = 0; i < disk_verts.size(); i++) {

		const auto &v2 = disk_verts[i];
		const auto &v1 = disk_verts[(i + 1) % disk_verts.size()];

		auto t = std::make_shared<Triangle>(v0, v1, v2);
		hemi_subdiv.push_back(t);
	}

	// subdiv
	for (int i = 0; i < 1; i++) {
		for (const auto& t : hemi_subdiv) {
			auto subdiv = midpoint_subdivide(t);
			hemi_tmp.insert(hemi_tmp.end(), subdiv.begin(), subdiv.end());
		}
		hemi_subdiv = hemi_tmp;
		hemi_tmp.clear();
	}

	// bloat
	glm::highp_vec3 dir_2 = glm::normalize(matrix * glm::vec4(orientation, 0.f));
	for (auto& t : hemi_subdiv) {
		bloat_vert(t->v0, v, dir_2);
		bloat_vert(t->v1, v, dir_2);
		bloat_vert(t->v2, v, dir_2);
	}

	triangles.insert(triangles.end(), hemi_subdiv.begin(), hemi_subdiv.end());
}

void Turtle::bloat_vert(Vertex& v, const TurtleVertex& tv, const glm::vec3 orientation)
{
	auto dir = glm::normalize(v.position - tv.pos);
	v.normal = dir;

	auto scaler = glm::max(glm::dot(dir, orientation), 0.f);

	auto a = tv.pos + dir * tv.thickness;
	auto b = a - v.position;

	auto foo = glm::length(b) * scaler;

	v.position = v.position + dir * foo;
}


//		    (v0)
//		     /\
//		    /  \
//		   /	\
//		  /      \
//   (v01)--------(v20)
//	    /\        /\
//	   /  \      /  \
// 	  /    \    /    \
//   /	    \  /      \
// (v1)-----(v12)-----(v2)

std::vector<std::shared_ptr<Triangle>> Turtle::midpoint_subdivide(std::shared_ptr<Triangle> t)
{
	std::vector<std::shared_ptr<Triangle>> out;

	auto v0 = t->v0;
	auto v01 = get_midway_point(t->v0, t->v1);
	auto v1 = t->v1;
	auto v12 = get_midway_point(t->v1, t->v2);
	auto v2 = t->v2;
	auto v20 = get_midway_point(t->v2, t->v0);

	out.push_back(std::make_shared<Triangle>(v0, v01, v20));
	out.push_back(std::make_shared<Triangle>(v01, v1, v12));
	out.push_back(std::make_shared<Triangle>(v20, v12, v2));
	out.push_back(std::make_shared<Triangle>(v01, v12, v20));

	return out;
}

Vertex Turtle::get_midway_point(Vertex v0, Vertex v1)
{
	auto dir = glm::normalize(v1.position - v0.position);
	auto halfway_dist = glm::distance(v0.position, v1.position) / 2.f;

	Vertex v;
	v.position = v0.position + dir * halfway_dist;

	return v;
}

void Turtle::gen_tube(int granularity, TurtleVertex start, TurtleVertex end)
{
	auto start_verts = gen_unit_disk(granularity, start);
	auto end_verts = gen_unit_disk(granularity, end);

	//gen_hemisphere(glm::vec3(0,-1,0), start_verts, start);
	//gen_hemisphere(glm::vec3(0,1,0), end_verts, end);
	
	for (int i = 0; i < granularity; i++)
	{
		Vertex v0 = start_verts[i];
		Vertex v1 = start_verts[(i + 1) % granularity];

		Vertex v2 = end_verts[i];
		Vertex v3 = end_verts[(i + 1) % granularity];

		triangles.push_back(std::make_shared<Triangle>(v0, v2, v1));
		triangles.push_back(std::make_shared<Triangle>(v2, v3, v1));
	}
}

void Turtle::update_orientation(glm::mat4 mat) {
	curr_state.dir = glm::normalize(mat * glm::vec4(curr_state.dir, 0.f));
	curr_state.right = glm::normalize(glm::cross(curr_state.dir, world_up));
	curr_state.up = glm::normalize(glm::cross(curr_state.right, curr_state.dir));
}

