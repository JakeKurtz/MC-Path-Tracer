#include "LRule.h"

LRule::LRule(std::string input)
{
	(this)->input = input;

	std::string symbol_raw = "", condition_raw = "";
	std::string replacement_raw = "", prob_raw = "";

	boost::smatch m;
	if (boost::regex_match(input, m, input_structure_re)) {
		boost::regex_search(input, m, boost::regex(std::string(m[2])));
		std::string prefix = m.prefix();
		std::string suffix = m.suffix();

		if (boost::regex_search(prefix, m, boost::regex(":"))) {
			symbol_raw = m.prefix();
			condition_raw = m.suffix();
		}
		else {
			symbol_raw = prefix;
			condition_raw = "1";
		}

		if (boost::regex_search(suffix, m, boost::regex(":"))) {
			replacement_raw = m.prefix();
			prob_raw = m.suffix();
		}
		else {
			replacement_raw = suffix;
			prob_raw = "1";
		}

		process_symbol_string(symbol_raw);
		process_conditional_string(condition_raw);
		process_replacement_string(replacement_raw);
		process_prob_string(prob_raw);
	}

	if (valid_symbols &&
		valid_conditional &&
		valid_replacement &&
		valid_probability) 
	{
		valid_rule = true;
	}
	else {
		std::cerr << "LRule Error: the rule format is invalid." << std::endl;
	}
}

std::string LRule::get_symbol()
{
	return symbol->symbol;
}

void LRule::load_variables()
{
	for (auto v : left_context->parameters) {
		symbol_table.create_variable(v);
		variables.insert(v);
	}

	for (auto v : symbol->parameters) {
		symbol_table.create_variable(v);
		variables.insert(v);
	}

	for (auto v : right_context->parameters) {
		symbol_table.create_variable(v);
		variables.insert(v);
	}

	condition_exp.register_symbol_table(symbol_table);
}

void LRule::process_symbol_string(std::string s)
{
	boost::regex re_0("<");
	boost::regex re_1(">");

	boost::smatch m;

	std::string lc, sym, rc;

	if (boost::regex_search(s, m, re_0)) {
		lc = m.prefix();
		sym = m.suffix();

		if (boost::regex_search(sym, m, re_1)) {
			rc = m.suffix();
			sym = m.prefix();
		}
	}
	else {
		if (boost::regex_search(s, m, re_1)) {
			rc = m.suffix();
			sym = m.prefix();
		}
		else {
			sym = s;
		}
	}

	left_context = new LModule(lc);
	symbol = new LModule(sym);
	right_context = new LModule(rc);

	if (verify_symbols()) {
		load_variables();
		valid_symbols = true;
	}
}

void LRule::process_conditional_string(std::string s)
{
	boost::smatch m;
	if (!boost::regex_match(s, m, empty_re)) {
		condition = s;
		boost::trim(condition);
	}
	else {
		condition = "1";
	}

	if (!parser.compile(condition, condition_exp))
	{
		std::cerr << "LRule Error: parser failed to compile. The conditional expression \"" << s << "\" is invalid." << std::endl;
	}
	else {
		valid_conditional = true;
	}
}

void LRule::process_replacement_string(std::string s)
{
	// NOTE: Ensure the replacement string is using the correct alphabet via regex.
	// If so, set the replacement string to s and contintue, otherwise exit.

	std::string suffix = "";
	std::string prefix = "";

	s = boost::regex_replace(s, boost::regex(" "), "");

	boost::smatch m;
	bool found_match = true;
	valid_replacement = true;
	while (boost::regex_search(s, m, parameters_re) && valid_replacement)
	{
		prefix = m.prefix();
		replacement += prefix + "(";

		std::string parameters = trim_brackets(m.str(0));
		std::stringstream ss(parameters);

		int i = 0;
		while (ss.good()) 
		{
			if (i > 0) {
				replacement += ",";
			}

			std::string p_str;
			getline(ss, p_str, ',');
			boost::trim(p_str);

			expression_t var_exp;
			var_exp.register_symbol_table(symbol_table);

			if (!parser.compile(p_str, var_exp)) {
				std::cerr << "LRule Error: the replacement parameter \"" << p_str << "\" failed to compile." << std::endl;
				valid_replacement = false;
				break;
			}

			replacement_parameters.insert(std::pair<std::string, expression_t>(p_str, var_exp));
			replacement += "<" + p_str + ">";

			i++;
		}

		replacement += ")";

		s = m.suffix().str();
	}
	replacement += s;
}

void LRule::process_prob_string(std::string s)
{
	float probability;

	try {
		probability = std::stof(s);
	}
	catch (const std::exception& e) {
		std::cerr << "LRule Error: the production probability " << s << " is not a number." << std::endl;
	}

	if (probability > 1.f || probability < 0.f) {
		std::cerr << "LRule Error: the production probability " << probability << " is outside the range [0, 1]." << std::endl;
	}
	else {
		(this)->prob = probability;
		valid_probability = true;
	}
}

void LRule::load_module_parameters(const LModule* lmodule, const boost::string_ref input_string)
{
	if (lmodule->parameters.size() > 0) {
		std::string parameters_raw = trim_brackets(input_string.substr(1, input_string.length()));
		std::stringstream ss(parameters_raw);
		int i = 0;
		while (ss.good() && i < lmodule->parameters.size())
		{
			std::string p_str;
			getline(ss, p_str, ',');

			std::string var = lmodule->parameters[i];

			try {
				symbol_table.get_variable(var)->ref() = std::stof(p_str);
			}
			catch (const std::exception& e) {
				std::cerr << "LRule Error: expected a variable for the module " << lmodule->symbol << ". Value is set to 0.0" << std::endl;
				symbol_table.get_variable(var)->ref() = 0.f;
			}

			i++;
		}
	}
}

bool LRule::verify_symbols()
{
	bool valid = true;
	if (!left_context->valid) {
		std::cout << "LRule Error: Failed to varify production. The left context \"" << left_context->symbol << "\" is not a valid module." << std::endl;
		valid = false;
	}

	if (!symbol->valid) {
		std::cout << "LRule Error: Failed to varify production. The symbol \"" << symbol->symbol << "\" is not a valid module." << std::endl;
		valid = false;
	}

	if (symbol->symbol == "") {
		std::cout << "LRule Error: Failed to varify production. The Symbol cannot be an empty string." << std::endl;
		valid = false;
	}

	if (!right_context->valid) {
		std::cout << "LRule Error: Failed to varify production. The right context \"" << right_context->symbol << "\" is not a valid module." << std::endl;
		valid = false;
	}

	return valid;
}

bool LRule::verify_context(const LModule* context, const boost::string_ref context_str)
{
	if (context->symbol == "") {
		return true;
	}
	else {
		return (std::string(1, context_str[0]) == context->symbol);
	}
}

bool LRule::verify_symbol_context(const boost::string_ref lc, const boost::string_ref rc) 
{
	return (verify_context(left_context, lc) && verify_context(right_context, rc));
}

bool LRule::valid_input_string(std::string input)
{
	boost::smatch m;
	return boost::regex_match(input, m, input_structure_re);
}

std::string LRule::get_input_str()
{
	return input;
}

bool LRule::apply(const boost::string_ref lc, const boost::string_ref sym, const boost::string_ref rc, std::string& output_string)
{
	output_string = std::string(sym);

	if (verify_symbol_context(lc, rc))
	{
		load_module_parameters(left_context, lc);
		load_module_parameters(symbol, sym);
		load_module_parameters(right_context, rc);

		std::string tmp_replace(replacement);

		if (static_cast<bool>(condition_exp.value()))
		{
			for (auto p : replacement_parameters)
			{
				std::string val = std::to_string(static_cast<float>(p.second.value()));
				boost::replace_all(tmp_replace, "<" + p.first + ">", val);
			}
			output_string = tmp_replace;
			return true;
		}
	}
	return false;
}

bool LRule::valid()
{
	return (this)->valid_rule;
}
