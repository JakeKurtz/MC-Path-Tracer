#include "LSystem.h"

LSystem::LSystem() {
	axiom = "a";
	l_system = axiom;
	franklin = new Turtle();
}

std::string LSystem::string()
{
	return l_system;
}

bool LSystem::add_rule(std::string rule) {

	LRule* r = new LRule(rule);

	if (r->valid()) {
		auto sym = r->get_symbol();

		if (rules.find(sym) == rules.end()) {
			std::vector<LRule*> v = { r };
			rules.insert(std::pair<std::string, std::vector<LRule*>>(sym, v));
		}
		else {
			rules[sym].push_back(r);
		}
		return true;
	}
	else {
		std::cerr << "LSystem Error: failed to add the rule \"" << rule << "\"." << std::endl;
		return false;
	}
}

bool LSystem::delete_rule(std::string sym, int index)
{
	auto pos = rules.find(sym);
	if (pos != rules.end()) {
		pos->second.erase(pos->second.begin() + index);

		if (pos->second.empty()) {
			rules.erase(sym);
		}
		return true;
	}
	else {
		return false;
	}
}

void LSystem::set_generations(int generations)
{
	(this)->generations = generations;
}

std::string LSystem::get_axiom()
{
	return axiom;
}

int LSystem::get_generations()
{
	return generations;
}

std::map<std::string, std::vector<LRule*>> LSystem::get_rules()
{
	return rules;
}

Turtle* LSystem::get_turtle()
{
	return franklin;
}

void LSystem::set_axiom(std::string axiom)
{
	(this)->axiom = axiom;
}

bool LSystem::build()
{
	l_system = axiom;
	for (int i = 0; i < generations; i++) {
		generate();
	}
	franklin->set_command_str(l_system);
	return franklin->build_curve();
}

void LSystem::generate()
{
	boost::string_ref l_sys_ref = l_system;

	boost::string_ref l_context = "";
	boost::string_ref symbol = "";
	boost::string_ref r_context = "";

	std::string output_string = "";
	std::string replacement = "";

	int i = 0;
	while (i < l_system.size()) {
		std::string sym = std::string(1, l_sys_ref[i]);

		int sym_start = i, sym_end = i;

		get_symbol_range(l_sys_ref, sym_start, sym_end);
		int sym_length = sym_end - sym_start;
		symbol = l_sys_ref.substr(sym_start, sym_length);

		replacement = std::string(symbol);

		int rc_start = sym_end, rc_end = sym_end;

		get_symbol_range(l_sys_ref, rc_start, rc_end);
		r_context = l_sys_ref.substr(rc_start, rc_end - rc_start);

		bool applied_rule = false;
		if (rules.find(sym) != rules.end()) {
			for (auto r : rules[sym]) {
				if (r->apply(l_context, symbol, r_context, replacement)) {
					output_string += replacement;
					applied_rule = true;
					break;
				}
			}
		}
		if (!applied_rule) {
			output_string += replacement;
		}
		l_context = symbol;
		i += sym_length;
	}
	l_system = output_string;
}

void LSystem::get_symbol_range(const boost::string_ref s, int& start_pos, int& end_pos)
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
