#include "RenderObject.h"

#include "Curve.h"
#include "Mesh.h"

int RenderObject::get_id()
{
    return id;
}

void RenderObject::set_name(std::string name)
{
    //(this)->name = name;
}

std::string RenderObject::get_name()
{
    return name;
}

void RenderObject::set_material(std::shared_ptr<Material> mat)
{
    (this)->material = mat;
}

std::shared_ptr<Material> RenderObject::get_material()
{
    return material;
}

void RenderObject::set_transform(shared_ptr<Transform> transform)
{
    (this)->transform = transform;
}

shared_ptr<Transform> RenderObject::get_transform()
{
    return transform;
}

int RenderObject::type()
{
    if (dynamic_cast<Curve*>(this) != nullptr)
    {
        return TYPE_CURVE;
    }
    else if (dynamic_cast<Mesh*>(this) != nullptr)
    {
        return TYPE_TRIANGLE_MESH;
    }
    else {
        //std::cout << "ERROR: invalid object type." << std::endl;
        return -1;
    }
}
