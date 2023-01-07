#pragma once

#include "Curve.h"

class Bezier : public Curve
{
public:
    Bezier();
    Bezier(std::vector<glm::vec3> control_points);
    ~Bezier();
    void update();
private:
    std::vector<glm::vec3> de_casteljau(std::vector<glm::vec3> const& points, int depth);
};

