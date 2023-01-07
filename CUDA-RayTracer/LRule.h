#pragma once

#include "LModule.h"
#include <exprtk.hpp>
#include <iostream>
#include <boost/algorithm/string.hpp>

typedef double T;
typedef exprtk::symbol_table<T> symbol_table_t;
typedef exprtk::expression<T>   expression_t;
typedef exprtk::parser<T>       parser_t;

class LRule {
public:

	LRule(std::string input);

	std::string get_symbol();

	std::string get_input_str();
	bool apply(const boost::string_ref left_context, const boost::string_ref symbol, const boost::string_ref right_context, std::string& replacement);
	bool valid();

private:
	LModule* left_context;
	LModule* symbol;
	LModule* right_context;

	std::string input;
	std::string condition;
	std::string replacement;
	std::set<std::string> variables;
	std::map<std::string, expression_t> replacement_parameters;

	float prob = 1.f;

	bool valid_symbols = false;
	bool valid_conditional = false;
	bool valid_replacement = false;
	bool valid_probability = false;
	bool valid_rule = false;

	symbol_table_t symbol_table;
	expression_t condition_exp;
	parser_t parser;

	void load_variables();

	void process_symbol_string(std::string s);
	void process_conditional_string(std::string s);
	void process_replacement_string(std::string s);

	bool verify_context(const LModule* context, const boost::string_ref context_str);
	bool verify_symbol_context(const boost::string_ref lc, const boost::string_ref rc);

	void process_prob_string(std::string s);

	void load_module_parameters(const LModule* lmodule, const boost::string_ref input_string);

	bool verify_symbols();

	bool valid_input_string(std::string input);
};
