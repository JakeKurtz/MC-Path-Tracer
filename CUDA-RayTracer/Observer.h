#pragma once
#include <string>

class Observer
{
public:
	virtual void update(const std::string& msg) = 0;
};

