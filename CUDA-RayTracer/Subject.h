#pragma once
#include "Observer.h"
#include <memory>
#include <list>

class Subject
{
public:
	~Subject() {};
	virtual void attach(const std::shared_ptr<Observer> observer);
	virtual void detach(const std::shared_ptr<Observer> observer);
protected:
	std::list<std::shared_ptr<Observer>> observers;
	virtual void notify(const std::string& msg) = 0;
};

