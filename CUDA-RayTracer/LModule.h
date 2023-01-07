#pragma once
#include <string>
#include <set>
#include <vector>
#include <regex>
#include <sstream>

#include <boost/regex.hpp>
#include <boost/utility/string_ref.hpp>

const boost::regex func_name_re(" *[A-Za-z]{1} *(\\(|)");
const boost::regex input_structure_re(".*(:|).*(=|\\->).*(:|)");
const boost::regex empty_re("( *|^$)");

const boost::regex valid_module_re("( *[A-Za-z]{1} *\\( *([A-Za-z]+(?: *, *[A-Za-z]+)+|[A-Za-z]) *\\) *| *[A-Za-z]{1} *)|^$");
const boost::regex valid_boolean_re("and|or|not|xor|<=|<|>|>=|!=|==|[<>!=]=|[<>]");
const boost::regex valid_math_ex("[ a-zA-Z0-9*+\\-*\\/%^]*");

const boost::regex parameters_re("\\((?:[^)(]+|\\((?:[^)(]+|\\([^)(]*\\))*\\))*\\)");
const boost::regex module_re("[A-Za-z]{1}(\\((?:[^)(]+|\\((?:[^)(]+|\\([^)(]*\\))*\\))*\\)|)");

struct LModule {
	std::string symbol = "";
	std::vector<std::string> parameters;

	bool valid = false;

	LModule(std::string input);

	std::string extract_symbol(std::string input);
	void extract_parameters(std::string input);
};

std::string trim_brackets(const boost::string_ref& s);
