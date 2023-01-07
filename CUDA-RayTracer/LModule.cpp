#include "LModule.h"

LModule::LModule(std::string input)
{
	boost::smatch m;
	if (!boost::regex_match(input, m, valid_module_re)) {
		symbol = input;
		valid = false;
	}
	else {
		symbol = extract_symbol(input);
		extract_parameters(input);
		valid = true;
	}
}

void LModule::extract_parameters(std::string input)
{
	boost::smatch m;
	while (boost::regex_search(input, m, parameters_re))
	{
		std::string paramerters_raw = trim_brackets(m.str(0));
		std::stringstream ss(paramerters_raw);

		while (ss.good())
		{
			std::string p_str;
			getline(ss, p_str, ',');
			parameters.push_back(p_str);
		}
		input = m.suffix().str();
	}
}

std::string LModule::extract_symbol(std::string input)
{
	std::string out = "";

	boost::smatch m;
	boost::regex_search(input, m, func_name_re);

	if (m.size() > 0) {
		std::string name_raw = m[0];
		out = std::regex_replace(name_raw, std::regex("[() ]"), "");
	}

	return out;
}

std::string trim_brackets(const boost::string_ref& s) {
	std::string temp = std::string(s);
	if (temp.size() > 0 && temp.front() == '(' && temp.back() == ')') {
		temp.erase(0, 1);
		temp.pop_back();
	}
	return temp;
};