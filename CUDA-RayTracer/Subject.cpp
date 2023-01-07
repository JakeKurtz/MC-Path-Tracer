#include "Subject.h"

void Subject::attach(const std::shared_ptr<Observer> observer)
{
	if (!(std::find(observers.begin(), observers.end(), observer) != observers.end())) {
		observers.push_back(observer);
	}
}

void Subject::detach(std::shared_ptr<Observer> observer)
{
	observers.remove(observer);
}
