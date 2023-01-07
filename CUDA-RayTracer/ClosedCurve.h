#pragma once

#include "Curve.h"

class ClosedCurve : public Curve
{
public:
    ClosedCurve();
    ClosedCurve(std::vector<glm::vec3> control_points);
    ~ClosedCurve();
    void update();
private:
    std::vector<glm::vec3> subdiv_closed(std::vector<glm::vec3> const& points, int degree, int depth);
};

