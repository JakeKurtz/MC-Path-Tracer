#include "Bezier.h"

Bezier::Bezier()
{
    id = gen_id();
    name = gen_object_name("Bezier");

    this->control_points.push_back(glm::vec3(0, 1, 0));
    this->control_points.push_back(glm::vec3(0, 0, 1));

    transform = std::make_shared<Transform>();
    material = std::make_shared<Material>();

    update();
}

Bezier::Bezier(std::vector<glm::vec3> control_points)
{
    id = gen_id();
    name = gen_object_name("Bezier");

    this->control_points = control_points;

    transform = std::make_shared<Transform>();
    material = std::make_shared<Material>();

    update();
}

Bezier::~Bezier()
{
    remove_object_name(name);
}

void Bezier::update()
{
    this->vertices = de_casteljau(control_points, depth);
    setup_buffers();
}

std::vector<glm::vec3> Bezier::de_casteljau(std::vector<glm::vec3> const& points, int depth)
{
    std::vector<glm::vec3> out, tmp, cp;
    out.push_back(points[0]);

    float t = 1.f / depth;

    for (int i = 1; i <= depth; i++) {
        cp = points;
        while (cp.size() != 1) {
            for (int j = 1; j < cp.size(); j++) {
                tmp.push_back(glm::mix(cp[j - 1], cp[j], t * i));
            }
            cp.swap(tmp);
            tmp.clear();
        }
        out.push_back(cp[0]);
    }

    return out;
}
