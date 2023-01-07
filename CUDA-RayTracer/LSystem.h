#pragma once

#include <boost/utility/string_ref.hpp>
#include <string>
#include <map>
#include <exprtk.hpp>
#include <regex>
#include <iostream>
#include "LRule.h"
#include "Turtle.h"

class LSystem
{
public:
	LSystem();

	std::string string();

	bool add_rule(std::string rule);
	bool delete_rule(std::string sym, int index);

	void set_axiom(std::string axiom);
	void set_generations(int generations);

	std::string get_axiom();
	int get_generations();
	std::map<std::string, std::vector<LRule*>> get_rules();

	Turtle* get_turtle();

	bool build();

private:
	std::string axiom;
	std::string l_system;
	std::map<std::string, std::vector<LRule*>> rules;

	Turtle* franklin;

	void generate();
	void get_symbol_range(const boost::string_ref s, int& start_pos, int& end_pos);

	int generations = 2;
};

