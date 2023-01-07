#include "wavefront_kernels.cuh"

surface<void, cudaSurfaceType2D> surf;
cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 0, cudaChannelFormatKindUnsigned);

__global__
void draw_to_surface(dCamera* camera, dFilm* film)
{
    const uint32_t id = get_thread_id();

    int nmb_pixels = film->width * film->height;

    if (id >= nmb_pixels)
        return;

    int x = id % film->width;
    int y = (id / film->width) % film->height;
    jek::Vec3f color = film->Ld[id] / film->samples[id];
    color *= film->exposure_time;
    color /= (color + 1.0f);

    /*
    auto lum = jek::luminance(color);
    if (lum < .15) {
        auto v = jek::remap(0.f, 0.15f, 0.f, 1.f, lum);
        color = jek::mix(jek::Vec3f(0, 0, 1), jek::Vec3f(0, 1, 0), v);
    }
    else if (.15 < lum && lum < .5) {
        auto v = jek::remap(0.15f, 0.5f, 0.f, 1.f, lum);
        color = jek::mix(jek::Vec3f(0, 1, 0), jek::Vec3f(1, 1, 0), v);
    }
    else {
        auto v = jek::remap(0.5f, 1.f, 0.f, 1.f, lum);
        color = jek::mix(jek::Vec3f(1, 1, 0), jek::Vec3f(1, 0, 0), v);
    }
    */

    char4 color_out = make_char4((unsigned int)(255 * color.x), (unsigned int)(255 * color.y), (unsigned int)(255 * color.z), 255);
    surf2Dwrite(color_out, surf, x * sizeof(char4), y, cudaBoundaryModeZero);
}

__global__
void wf_init(
    Paths* paths,
    uint32_t path_pool_size)
{
    uint32_t id = get_thread_id();

    if (id >= path_pool_size)
        return;

    paths->dead[id] = true;
}

__global__
void g_clear_dfilm(dFilm* film)
{
    uint32_t id = get_thread_id();

    if (id >= film->width*film->height)
        return;

    film->paths->dead[id] = true;
    film->samples[id] = 0;
    film->Ld[id] = jek::Vec3f(0.f);
}

void clear_dfilm(dFilm* film)
{
    uint32_t block_size = 256;
    uint32_t num_blocks = (film->width * film->height + block_size - 1) / block_size;

    g_clear_dfilm <<< num_blocks, block_size >>> (film);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void wavefront_init(
    Paths* paths, 
    uint32_t nmb_paths)
{
    uint32_t block_size = 256;
    uint32_t num_blocks = (nmb_paths + block_size - 1) / block_size;

    wf_init <<< num_blocks, block_size >>> (paths, nmb_paths);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__
void wf_logic(
    Queues* queues,
    dScene* scene,
    dCamera* camera,
    dFilm* film)
{
    uint32_t thrd_id = get_thread_id();

    if (thrd_id >= film->tile_width * film->tile_height)
        return;

    //uint32_t x, y;
    //film->get_tile_pos(thrd_id, x, y);

    int x = (thrd_id % film->tile_width) + (film->tile_x_pos * film->tile_width);
    int y = ((thrd_id / film->tile_width) % film->tile_height) + (film->tile_y_pos * film->tile_height);

    uint32_t path_id = y * film->width + x;

    if (x >= film->width - 1 || y >= film->height - 1) return;

    Paths* paths = film->paths;

    uint32_t            pixel_id = path_id;//paths->pixel_id[path_id];
    dLight**            lights = scene->lights;
    int                 nmb_lights = scene->nmb_lights;
    dEnvironmentLight*  environment_light = scene->environment_light;
    jek::Vec3f          beta = paths->beta[path_id];
    uint32_t            path_length = paths->len[path_id];

    Isect isect = paths->isect[path_id];
    dRay ray = paths->ray[path_id];

    if (!paths->dead[path_id] && film->samples[pixel_id] < 250)
    {
        bool terminate = false;

        // Draw background
        if (path_length == 1)
        {
            if (isect.was_found) {
                film->Ld[pixel_id] += jek::Vec3f(0.f) * beta; // Note: zero for now, but intended for radiant objects.
            }
            else {
                for (int i = 0; i < scene->nmb_lights; i++) {
                    //film->Ld[pixel_id] += scene->lights[i]->L(isect, ray.d) * beta;
                    film->Ld[pixel_id] += scene->environment_light->L(isect, ray.d) * beta;
                }
            }
        }

        if (path_length > 5 || !isect.was_found)
        {
            terminate = true;
            //goto TERMINATE;
        }

        if (path_length > 5) goto TERMINATE;

        if (path_length > 1)
        {
            jek::Vec3f  f_light     = paths->f_light[path_id];
            jek::Vec3f  f_brdf      = paths->f_brdf[path_id];
            jek::Vec3f  f_sample    = paths->f_sample[path_id];

            jek::Vec3f  Li_light    = paths->Li_light[path_id];
            jek::Vec3f  Li_brdf     = paths->Li_brdf[path_id];

            jek::Vec2f  pdf_light   = paths->pdf_light[path_id];
            jek::Vec2f  pdf_brdf    = paths->pdf_brdf[path_id];
            float       pdf_sample  = paths->pdf_sample[path_id];

            float       visible     = paths->visible[path_id];

            jek::Vec3f  Ld;
            float       weight;

            // LIGHT SAMPLE
            weight = power_heuristic(1, pdf_light.x, 1, pdf_brdf.y);
            if (weight > 0.f && visible && pdf_light.x > 0.f) {
                Ld += f_light * Li_light * weight / pdf_light.x;
            }

            // BRDF SAMPLE
            weight = power_heuristic(1, pdf_brdf.x, 1, pdf_light.y);
            if (weight > 0.f && pdf_brdf.x > 0.f) {
                Ld += f_brdf * Li_brdf * weight / pdf_brdf.x;
            }

            film->Ld[pixel_id] += Ld * beta;

            if (f_sample == jek::Vec3f(0.f) || pdf_sample == 0.f) {
                terminate = true;
                goto TERMINATE;
            }

            paths->beta[path_id] *= f_sample / pdf_sample;

            if (path_length > 3) {
                float q = fmaxf((float).05, 1.f - beta.y);
                if (jek::rand_float() < q) {
                    terminate = true;
                    goto TERMINATE;
                }
                beta /= 1.f - q;
            }
        }

    TERMINATE:
        if (terminate)
        {
            paths->dead[path_id] = true;
            film->samples[pixel_id]++;
        }
        else // path continues
        {
            jek::Vec3f lightdir;

            auto l_id = jek::rand_int(0, nmb_lights);

            paths->light_id[path_id] = (l_id == nmb_lights) ? 0 : l_id;
            lights[paths->light_id[path_id]]->dir(paths->isect[path_id], lightdir);
            paths->ray_light[path_id] = dRay(paths->isect[path_id].position + paths->isect[path_id].normal * 0.01f, lightdir);

            queues->mat_mix_path[atomicAdd(&queues->mat_mix_len, 1)] = path_id;
        }
    }

    if (paths->dead[path_id] && film->samples[pixel_id] < film->nmb_samples)
    {
        queues->new_path[atomicAdd(&queues->new_path_len, 1)] = path_id;
    }
}

__global__
void wf_generate(
    Queues* queues,
    dScene* scene,
    dCamera* camera,
    dFilm* film)
{
    uint32_t thrd_id = get_thread_id();
    if (thrd_id >= queues->new_path_len)
        return;

    uint32_t id = queues->new_path[thrd_id];
    Paths* paths = film->paths;

    //uint32_t x, y;
    //film->get_pos(id, x, y);

    int x = id % film->width;
    int y = (id / film->width) % film->height;
    
    paths->dead[id] = false;
    paths->ray[id]  = camera->gen_ray(film, x, y);
    paths->len[id]  = 0;
    paths->beta[id] = jek::Vec3f(1.f);

    queues->ext_path[atomicAdd(&queues->ext_path_len, 1)] = id;
}

__global__
void wf_extend(
    Queues* queues,
    dScene* scene,
    dFilm* film)
{
    uint32_t thrd_id = get_thread_id();
    if (thrd_id >= queues->ext_path_len)
        return;

    uint32_t id = queues->ext_path[thrd_id];
    Paths* paths = film->paths;

    Isect isect;
    dRay ray = paths->ray[id];

    intersect(scene->nodes, scene->triangles, ray, isect);
    paths->len[id]++;
    paths->isect[id] = isect;
}

__global__
void wf_shadow(
    Queues* queues,
    dScene* scene,
    dFilm* film)
{
    uint32_t thrd_id = get_thread_id();
    if (thrd_id >= queues->shadow_len)
        return;

    uint32_t id = queues->shadow_path[thrd_id];
    Paths* paths = film->paths;

    dRay ray = paths->ray_light[id];
    int light_id = paths->light_id[id];

    if (light_id != -1) {
        paths->visible[id] = scene->lights[light_id]->visible(scene->nodes, scene->triangles, ray);
    }
}

__global__
void wf_mat_mix(
    Queues* queues,
    dScene* scene,
    dFilm* film)
{
    uint32_t thrd_id = get_thread_id();

    if (thrd_id >= queues->mat_mix_len)
        return;

    uint32_t id = queues->mat_mix_path[thrd_id];
    Paths* paths = film->paths;

    jek::Vec3f f_light, Li_light;
    jek::Vec3f f_brdf, Li_brdf;
    jek::Vec2f pdf_light = jek::Vec2f(1), pdf_brdf = jek::Vec2f(1);
    jek::Vec3f wi_brdf;

    Isect       isect       =  paths->isect[id];
    uint32_t    light_id    =  paths->light_id[id];
    jek::Vec3f  light_wi    =  paths->ray_light[id].d;
    jek::Vec3f  wo          = -paths->ray[id].d;
    dLight*     light       =  scene->lights[light_id];

    // Light Sample

    //f_light = bsdf->f(isect, wo, light_wi);
    //Li_light = light->L(isect, light_wi);
    //pdf_light.x = light->pdf(isect, light_wi);
    //pdf_brdf.y = (!light->is_delta()) ? bsdf->pdf(isect, light_wi) : 1.f;
    f_light = spec_get_f(isect, light_wi, wo) + diff_get_f(isect, light_wi, wo);
    Li_light = light->L(isect, light_wi);
    pdf_light.x = light->pdf(isect, light_wi);
    pdf_brdf.y = (!light->is_delta()) ? (diff_get_pdf(isect, light_wi, wo) + spec_get_pdf(isect, light_wi, wo)) * 0.5f : 1.f;
     
    // BRDF Sample
    if (!light->is_delta()) {
        wi_brdf = (jek::rand_float() < 0.5) ? spec_get_wi(isect, wo) : diff_get_wi(isect);
        dRay vis_ray = dRay(isect.position + wi_brdf * 0.001f, wi_brdf);

        if (light->visible(scene->nodes, scene->triangles, vis_ray)) {
            //f_light = bsdf->sample_f(isect, wo, wi_brdf, pdf_brdf.x);
            //Li_light = light->L(isect, wi_brdf);
            //pdf_light.x = light->pdf(isect, wi_brdf);
            f_brdf = spec_get_f(isect, wi_brdf, wo) + diff_get_f(isect, wi_brdf, wo);
            Li_brdf = light->L(isect, wi_brdf);
            pdf_brdf.x = (diff_get_pdf(isect, wi_brdf, wo) + spec_get_pdf(isect, wi_brdf, wo)) * 0.5f;
            pdf_light.y = light->pdf(isect, wi_brdf);
        }
    }

    // Light Path Sample
    jek::Vec3f f_sample;
    jek::Vec3f wi_sample; 
    float pdf_sample;
    //f_sample = bsdf->sample_f(isect, wo, wi_sample, pdf_sample);
    {
        wi_sample = (jek::rand_float() < 0.5) ? spec_get_wi(isect, wo) : diff_get_wi(isect);
        pdf_sample = (diff_get_pdf(isect, wi_sample, wo) + spec_get_pdf(isect, wi_sample, wo)) * 0.5f;
        f_sample = spec_get_f(isect, wi_sample, wo) + diff_get_f(isect, wi_sample, wo);
    }

    dRay ray = dRay(isect.position + get_normal(isect) * 0.001f, wi_sample);

    paths->Li_light[id]     = Li_light;
    paths->Li_brdf[id]      = Li_brdf;

    paths->f_light[id]      = f_light;
    paths->f_brdf[id]       = f_brdf;
    paths->f_sample[id]     = f_sample;

    paths->pdf_light[id]    = pdf_light;
    paths->pdf_brdf[id]     = pdf_brdf;
    paths->pdf_sample[id]   = pdf_sample;

    paths->ray[id]          = ray;

    queues->ext_path[atomicAdd(&queues->ext_path_len, 1)] = id;
    queues->shadow_path[atomicAdd(&queues->shadow_len, 1)] = id;
}

void wavefront_pathtrace(
    std::shared_ptr<Scene> scene,
    std::shared_ptr<Camera> camera,
    std::shared_ptr<Film> film,
    Queues* queues,
    cudaArray_const_t array,
    cudaEvent_t event,
    cudaStream_t stream)
{
    uint32_t tile_width, tile_height;
    film->get_size(tile_width, tile_height);

    uint32_t block_size, num_blocks;

    block_size = 256;
    num_blocks = max((uint32_t)1, ((tile_width * tile_height) + block_size - 1) / block_size);
    wf_logic <<< num_blocks, block_size, 0, stream >>> (
        queues,
        scene->get_dptr(),
        camera->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaStreamSynchronize(stream));

    block_size = 256;
    num_blocks = max((uint32_t)1, (queues->new_path_len + block_size - 1) / block_size);
    wf_generate <<< num_blocks, block_size, 0, stream >>> (
        queues,
        scene->get_dptr(),
        camera->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaStreamSynchronize(stream));

    block_size = 256;
    num_blocks = max((uint32_t)1, (queues->mat_mix_len + block_size - 1) / block_size);
    wf_mat_mix <<< num_blocks, block_size, 0, stream >>> (
        queues,
        scene->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaStreamSynchronize(stream));

    block_size = 256;
    num_blocks = max((uint32_t)1, (queues->ext_path_len + block_size - 1) / block_size);
    wf_extend << < num_blocks, block_size, 0, stream >> > (
        queues,
        scene->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaStreamSynchronize(stream));

    block_size = 256;
    num_blocks = max((uint32_t)1, (queues->shadow_len + block_size - 1) / block_size);
    wf_shadow << < num_blocks, block_size, 0, stream >> > (
        queues,
        scene->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaStreamSynchronize(stream));
    
    cudaError_t cuda_err = cudaBindSurfaceToArray(surf, array);

    block_size = 256;
    num_blocks = ((tile_width * tile_height) + block_size - 1) / block_size;
    draw_to_surface <<< num_blocks, block_size, 0, stream >>> (
        camera->get_dptr(), 
        film->get_dptr());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__
void debug_kernel(
    dScene* scene,
    dCamera* camera,
    dFilm* film)
{
    uint32_t thread_id = get_thread_id();

    if (thread_id >= film->tile_width * film->tile_height)
        return;

    int x = (thread_id % film->tile_width) + (film->tile_x_pos * film->tile_width);
    int y = ((thread_id / film->tile_width) % film->tile_height) + (film->tile_y_pos * film->tile_height);

    uint32_t path_id = y * film->width + x;

    if (x >= film->width - 1 || y >= film->height - 1) return;

    dRay ray = camera->gen_ray(film, x, y);

    Isect isect;
    intersect(scene->nodes, scene->triangles, ray, isect);

    jek::Vec3f color = jek::Vec3f(1.f);

    jek::Vec3f sp;
    jek::Vec3f wo = ray.d;
    jek::Vec3f wi;

    
    //jek::Vec2f uv = jek::Vec2f((float)x / (float)film->width, (float)y / (float)film->height);
    //jek::Vec3f sd = jek::sample_spherical_direction(uv);
    //uv = jek::sample_spherical_map(sd);
    
    
    scene->environment_light->dir(isect, wi);
    jek::Vec2f uv = jek::sample_spherical_map(wi);

    uint32_t _x = uv.x * film->width;
    uint32_t _y = uv.y * film->height;

    //if (_x >= film->width - 1 || _y >= film->height - 1) return;

    uint32_t pix_id = _y * film->width + _x;
    
    /*
    if (isect.was_found) {
        jek::Vec3f ext_dir;
        jek::Vec3f ext_f;
        float ext_pdf;

        //dLight* light = scene->environment_light;
        //light->dir(isect, wi);

        //color = BRDF_L(light, isect, wi, -wo);

    }
    else {
        dEnvironmentLight* el = scene->environment_light;
        color = el->L(isect, wo);
    }
    */
    //lerp(make_jek::Vec3f(0.f,1.f,0.f), make_jek::Vec3f(1.f,0.f,0.f), color.x);

    //color_buffer[path_id] = make_jek::Vec3f((float)x/(float)camera->width, (float)y / (float)camera->height, 0.f);//color;
    //color_buffer[id] = color2;////make_jek::Vec3f((float)x/(float)camera->width, (float)y / (float)camera->height, 0.f);//color;
    //color_buffer[id] = color;//make_jek::Vec3f(color);

    //film->Ld[pix_id] += scene->environment_light->L(isect, wi) / scene->environment_light->pdf(isect, wi);
    //uv.y = 1.0 - uv.y;

    //float pdf;
    //surf2Dread(&pdf, scene->environment_light->pdf_texture, (int)(uv.x * (scene->environment_light->tex_width - 1)) * sizeof(float), (int)(uv.y * (scene->environment_light->tex_height - 1)));
    
    film->Ld[pix_id] += jek::luminance(scene->environment_light->L(isect, wi)) * 0.01f;
    film->samples[pix_id] = 1;
    //compl_pixels[rand_idx] = true;

}

void debug_raytracer(
    std::shared_ptr<Scene> s, 
    std::shared_ptr<Camera> camera,
    std::shared_ptr<Film> film,
    cudaArray_const_t array, 
    cudaEvent_t event, 
    cudaStream_t stream)
{
    uint32_t tile_width, tile_height;
    film->get_size(tile_width, tile_height);

    uint32_t block_size = 256;
    uint32_t num_blocks = max((uint32_t)1, ((tile_width * tile_height) + block_size - 1) / block_size);
    debug_kernel <<< num_blocks, block_size, 0, stream >>> (
        s->get_dptr(),
        camera->get_dptr(),
        film->get_dptr());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaError_t cuda_err = cudaBindSurfaceToArray(surf, array);

    block_size = 256;
    num_blocks = ((tile_width * tile_height) + block_size - 1) / block_size;
    draw_to_surface << < num_blocks, block_size, 0, stream >> > (
        camera->get_dptr(),
        film->get_dptr());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(stream));
}