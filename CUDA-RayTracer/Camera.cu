#include "Camera.h"
#include "dMath.cuh"
#include <glm/gtx/string_cast.hpp>
#include "wavefront_kernels.cuh"
#include "Sample.h"

typedef thrust::device_vector<int>::iterator dintiter;

struct is_true
{
    __host__ __device__
        bool operator()(const int x)
    {
        return (x == 1);
    }
};

dRay dCamera::gen_ray(const dFilm* film, const float x, const float y)
{
    jek::Vec2f pNDC; // [-1, 1] x [-1, 1]
    pNDC.x = 2.f * ((x + 0.5) / (float)film->width) - 1.f;
    pNDC.y = 1.f - 2.f * ((y + 0.5) / (float)film->height);

    jek::Vec4f pNearNDC = inv_view_proj_mat * jek::Vec4f(pNDC.x, pNDC.y, -1.f, 1.f);
    jek::Vec4f pFarNDC = inv_view_proj_mat * jek::Vec4f(pNDC.x, pNDC.y, 1.f, 1.f);

    jek::Vec3f pNear = jek::Vec3f(pNearNDC.x, pNearNDC.y, pNearNDC.z) / pNearNDC.w;
    jek::Vec3f pFar = jek::Vec3f(pFarNDC.x, pFarNDC.y, pFarNDC.z) / pFarNDC.w;

    dRay ray;
    ray.o = pNear;
    ray.d = normalize(pFar - pNear);

    if (lens_radius > 0.f) {
        jek::Vec3f pFocal = ray.o + ray.d * f;
        jek::Vec2f pLensNorm = jek::concentric_sample_disk() * lens_radius;

        jek::Vec4f pLensTrans = inv_view_mat * jek::Vec4f(pLensNorm.x, pLensNorm.y, 0.f, 1.f);
        jek::Vec3f pLens = jek::Vec3f(pLensTrans.x, pLensTrans.y, pLensTrans.z) / pLensTrans.w;

        ray.o = pLens;
        ray.d = normalize(pFocal - ray.o);
    }
    return ray;
}

Camera::Camera()
{
    id = gen_id();
    init_dptr();
}
Camera::~Camera()
{
    cudaFree(d_camera);
}

void Camera::init_dptr()
{
    checkCudaErrors(cudaMallocManaged(&d_camera, sizeof(dCamera)));
}

void Camera::move(const Camera_Movement direction, const float deltaTime)
{
    float velocity = movement_speed * deltaTime;
    if (direction == FORWARD)
        position += front * velocity;
    if (direction == BACKWARD)
        position -= front * velocity;
    if (direction == LEFT)
        position -= right * velocity;
    if (direction == RIGHT)
        position += right * velocity;
    if (direction == UP)
        position += up * velocity;
    if (direction == DOWN)
        position -= up * velocity;

    update();
}
void Camera::rotate(const float yaw, const float pitch)
{
    (this)->yaw += yaw * look_sensitivity;
    (this)->pitch += pitch * look_sensitivity;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    //if (constrainPitch)
    {
        if ((this)->pitch > 89.0f)
            (this)->pitch = 89.0f;
        if ((this)->pitch < -89.0f)
            (this)->pitch = -89.0f;
    }
    update();
}

void Camera::set_position(const glm::vec3 position)
{
    (this)->position = position;
    update();
}
void Camera::set_yaw_pitch(const float yaw, const float pitch)
{
    (this)->yaw = yaw;
    (this)->pitch = pitch;
    update();
}
void Camera::set_zoom(const float zoom)
{
    (this)->zoom = zoom;
    update();
}
void Camera::set_focal_distance(const float focal_distance)
{
    (this)->focal_distance = focal_distance;
    update();
}
void Camera::set_lens_radius(const float lens_radius)
{
    (this)->lens_radius = lens_radius;
    update();
}

void Camera::set_aspect_ratio(const float aspect_ratio)
{
    (this)->aspect_ratio = aspect_ratio;
    update();
}

glm::vec3 Camera::get_position() const
{
    return position;
}
void Camera::get_yaw_pitch(float& yaw, float& pitch) const
{

}
float Camera::get_zoom() const
{
    return zoom;
}
float Camera::get_focal_distance() const
{
    return focal_distance;
}
float Camera::get_lens_radius() const
{
    return lens_radius;
}

float Camera::get_aspect_ratio() const
{
    return aspect_ratio;
}

uint32_t Camera::get_id() const
{
    return id;
}
dCamera* Camera::get_dptr() const
{
    return d_camera;
}

glm::mat4 Camera::get_view_mat() const
{
    return lookat_mat;
}
glm::mat4 Camera::get_proj_mat() const
{
    return proj_mat;
}

void Camera::send_uniforms(Shader& shader) const
{
    shader.setMat4("projection", proj_mat);
    shader.setMat4("view", lookat_mat);
    shader.setVec3("camPos", position);
    shader.setVec3("camDir", front);
    shader.setVec3("camUp", up);
    shader.setVec3("camRight", right);
    shader.setFloat("cam_exposure", exposure);
}
void Camera::send_uniforms2(Shader& shader) const
{
    shader.setMat4("projection", proj_mat);
    shader.setMat4("view", glm::mat4(glm::mat3(lookat_mat)));
    shader.setVec3("camPos", position);
    shader.setVec3("camDir", front);
    shader.setVec3("camUp", up);
    shader.setVec3("camRight", right);
    shader.setFloat("cam_exposure", exposure);
}

void Camera::update()
{
    update_view_mat();
    update_proj_mat();

    d_camera->position = position;
    d_camera->zoom = zoom;
    d_camera->lens_radius = lens_radius;
    d_camera->f = focal_distance;
    d_camera->d = 1.f;
    d_camera->inv_view_proj_mat = glm::inverse(proj_mat * lookat_mat);
    d_camera->inv_view_mat = glm::inverse(lookat_mat);

    notify("");
}

void Camera::update_view_mat()
{
    // calculate the new Front vector
    glm::vec3 _front;
    _front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    _front.y = sin(glm::radians(pitch));
    _front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(_front);

    // also re-calculate the Right and Up vector
    right = glm::normalize(glm::cross(front, worldUp));
    up = glm::normalize(glm::cross(right, front));

    lookat_mat = glm::lookAt(position, position + front, up);
}

void Camera::notify(const std::string& msg)
{
    for (auto const& o : observers)
    {
        o->update(msg);
    }
}
