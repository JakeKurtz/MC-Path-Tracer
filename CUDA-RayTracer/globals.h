#pragma once
#include <vector>
#include <string>
#include <map>
#include <unordered_set>

extern bool GLFW_INIT;

extern std::size_t id_counter;

extern int global_mesh_count;

extern std::unordered_set<std::string> global_obj_names;

static std::size_t gen_id()
{
	return id_counter++;
}

static std::string build_name_string(std::string name, unsigned int count)
{
	std::string num(std::to_string(count));
	while (num.length() < 3) {
		num = "0" + num;
	}
	return name + "." + num;
};

static std::string add_object_name(std::string name, std::string base, unsigned int count)
{
	if (global_obj_names.insert(name).second) {
		return name;
	}
	else {
		count++;
		return add_object_name(build_name_string(base, count), base, count);
	}
}

static void remove_object_name(std::string name)
{
	global_obj_names.erase(name);
}

static std::string gen_object_name(std::string name) 
{
	return add_object_name(name, name, 0);
}