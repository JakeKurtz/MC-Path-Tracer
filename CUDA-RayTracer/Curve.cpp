#include "Curve.h"

Curve::Curve()
{
    name = gen_object_name("Curve");
    this->control_points.push_back(glm::vec3(0, 1, 0));
    this->control_points.push_back(glm::vec3(0, 0, 1));

    update();
}

Curve::Curve(std::vector<glm::vec3> control_points)
{
    name = gen_object_name("Curve");
    this->control_points = control_points;

    update();
}

glm::vec3 Curve::center_of_mass()
{
    return glm::vec3(0.f);
}

void Curve::draw(Shader& shader)
{
    material->send_uniforms(shader);
    
    // connect control points
    glBindVertexArray(VAO_cp);
    glDrawArrays(GL_LINE_STRIP, 0, control_points.size());
    glBindVertexArray(0);

    // draw control points
    glPointSize(6);
    glBindVertexArray(VAO_cp);
    glDrawArrays(GL_POINTS, 0, control_points.size());
    glBindVertexArray(0);

    // draw curve
    glLineWidth(2.f);
    glBindVertexArray(VAO_curve);
    glDrawArrays(GL_LINE_STRIP, 0, vertices.size());
    glBindVertexArray(0);
}

void Curve::set_depth(int depth)
{
    if (depth <= 0) {
        std::cout << "Cruve resolution can not be less than or equal to zero!\n";
        this->depth = 1;
    }
    else {
        this->depth = depth;
    }
}

void Curve::set_vertices(std::vector<glm::vec3> vertices)
{
    (this)->vertices = vertices;
}

void Curve::set_control_points(std::vector<glm::vec3> control_points)
{
    (this)->control_points = control_points;
}

void Curve::set_indices(std::vector<unsigned int> indices)
{
    (this)->indices = indices;
}

int Curve::get_depth()
{
    return depth;
}

void Curve::center()
{
    glm::vec3 min = glm::vec3(10000000000000000);
    glm::vec3 max = glm::vec3(-10000000000000000);

    for (auto v : control_points) {
        if (v.x > max.x || v.y > max.y || v.z > max.z) {
            max = v;
        }

        if (v.x < min.x || v.y > min.y || v.z > min.z) {
            min = v;
        }
    }

    //p = glm::vec3(-(max.x + min.x) / 2.0f, -(max.y + min.y) / 2.0f, -(max.z + min.z) / 2.0f);
    //model_mat = glm::translate(model_mat, position);
}

void Curve::update()
{
    this->vertices = subdiv_open(control_points, degree, depth);
    setup_buffers();
}

std::vector<glm::vec3> Curve::subdiv_open(std::vector<glm::vec3> const& points, int degree, int depth) {
    std::vector<glm::vec3> out;

    out.push_back(points[0]);

    // add midpoints
    for (int i = 0; i < points.size(); i++) {
        if (i == points.size() - 1) {
            out.push_back(points[i]);
        }
        else {
            glm::vec3 a = points[i];
            glm::vec3 b = points[i + 1];

            glm::vec3 midPoint = a + 0.5f * (b - a);
            out.push_back(a);
            out.push_back(midPoint);
        }
    }
    // chase
    for (int i = 0; i < degree - 1; i++) {
        std::vector<glm::vec3> temp = out;
        for (int j = 0; j < temp.size(); j++) {
            if (j == 0 || j == temp.size() - 1) {
                continue;
            }
            else {
                glm::vec3 a = temp[j];
                glm::vec3 b = temp[j + 1];

                glm::vec3 midPoint = a + 0.5f * (b - a);
                out[j] = midPoint;
            }
        }
    }

    if (depth > 1) out = subdiv_open(out, degree, depth -= 1);
    return out;
}

void Curve::setup_buffers()
{    
    // ------ Control Points ------ //

    // create buffers/arrays
    glGenVertexArrays(1, &VAO_cp);
    glGenBuffers(1, &VBO_cp);

    glBindVertexArray(VAO_cp);

    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO_cp);
    glBufferData(GL_ARRAY_BUFFER, control_points.size() * sizeof(glm::vec3), &control_points[0], GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glBindVertexArray(0);

    // ------ Curve ------ //

    // create buffers/arrays
    glGenVertexArrays(1, &VAO_curve);
    glGenBuffers(1, &VBO_curve);

    glBindVertexArray(VAO_curve);

    // load data into vertex buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO_curve);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

    // set the vertex attribute pointers
    // vertex Positions
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glBindVertexArray(0);
}
