#include "PerspectiveCamera.h"
#include "globals.h"
#include "massert.h"

PerspectiveCamera::PerspectiveCamera() : Camera()
{
    zoom = 90.f;
    znear = 1.f;
    zfar = 100.f;

    id = gen_id();

    update();
}

PerspectiveCamera::PerspectiveCamera(float _yfov, float _znear, float _zfar) : Camera()
{
    m_assert(isnan(_yfov), "_yfov is not a number!");
    m_assert(isnan(_znear), "_znear is not a number!");
    m_assert(isnan(_zfar), "_zfar is not a number!");

    zoom = _yfov;
    znear = _znear;
    zfar = _zfar;

    id = gen_id();

    update();
}

PerspectiveCamera::PerspectiveCamera(glm::vec3 _position, float _yfov, float _znear, float _zfar)
{
    m_assert(!isnan(_yfov), "_yfov is not a number!");
    m_assert(!isnan(_znear), "_znear is not a number!");
    m_assert(!isnan(_zfar), "_zfar is not a number!");

    position = _position;
    zoom = _yfov;
    znear = _znear;
    zfar = _zfar;

    id = gen_id();

    update();
}

void PerspectiveCamera::update_proj_mat()
{
    proj_mat = glm::perspective(zoom, aspect_ratio, znear, zfar);
}