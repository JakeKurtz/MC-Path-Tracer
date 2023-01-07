#pragma once
#include "GLCommon.h"

#include <string>
#include "Shader.h"
#include "globals.h"

#include <cuda_runtime.h>

#include "dRay.cuh"
#include "Film.h"
#include "Subject.h"

#include <vector>
#include <numeric>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include "Matrix.h"

enum Camera_Movement
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

struct dCamera
{
    jek::Vec3f position = jek::Vec3f(0, 0, 0);
    jek::Matrix4x4f inv_view_proj_mat;
    jek::Matrix4x4f inv_view_mat;

    float lens_radius = 1.f;	// lens radius
    float d = 100.f;			// view plane distance
    float f = 1.f;				// focal plane distance
    float zoom = 1.f;			// zoom factor

    __device__ dRay gen_ray(const dFilm* film, const float x, const float y);
};

class Camera : public Subject
{
public:
    Camera();
    ~Camera();

    void init_dptr();

    void move(const Camera_Movement direction, const float deltaTime);
    void rotate(const float yaw, const float pitch);

    void set_position(const glm::vec3 position);
    void set_yaw_pitch(const float yaw, const float pitch);
    void set_zoom(const float zoom);
    void set_focal_distance(const float focal_distance);
    void set_lens_radius(const float lens_radius);
    void set_aspect_ratio(const float aspect_ratio);

    glm::vec3 get_position() const;
    void get_yaw_pitch(float& yaw, float& pitch) const;
    float get_zoom() const;
    float get_focal_distance() const;
    float get_lens_radius() const;
    float get_aspect_ratio() const;

    uint32_t get_id() const;
    dCamera* get_dptr() const;

    glm::mat4 get_view_mat() const;
    glm::mat4 get_proj_mat() const;

    void send_uniforms(Shader& shader) const;
    void send_uniforms2(Shader& shader) const;

    void update();

private:
    dCamera* d_camera;

    virtual void update_proj_mat() = 0;
    void update_view_mat();
    void notify(const std::string& msg);

protected:
    std::string name;
    uint32_t id;

    float zfar;
    float znear;

    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp = glm::vec3(0.f, 1.f, 0.f);

    glm::mat4 lookat_mat;
    glm::mat4 proj_mat;

    float yaw = -90.f;
    float pitch = 0.f;

    float movement_speed = 2.5f;
    float look_sensitivity = 0.1f;
    float zoom = 0.f;
    float exposure = 1.f;

    float lens_radius = 0.0001f;
    float focal_distance = 35.f;
    float d = 50.f;

    float aspect_ratio = 1.f;
};

