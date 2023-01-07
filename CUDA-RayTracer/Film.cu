#include "Film.h"
#include "CudaHelpers.h"
#include "wavefront_kernels.cuh"

Film::Film()
{
	(this)->id = gen_id();
	(this)->width = 1;
	(this)->height = 1;

	(this)->nmb_samples = 1000;
	(this)->max_path_length = 5;

	init_film_tex();
	init_dptr();

	set_tile_size(256, 256);
}

void Film::set_exposure(const float exposure)
{
	(this)->exposure = exposure;
	dptr->exposure_time = exposure;
}
void Film::set_size(const uint32_t width, const uint32_t height)
{
	(this)->width = width;
	(this)->height = height;

	update_tex_size();
	update_dptr_size();
	update_tile_info();
	update_path_size();

	clear();
}
void Film::set_tile_size(const uint32_t width, const uint32_t height)
{
	tile_width = width;
	tile_height = height;

	dptr->tile_width = width;
	dptr->tile_height = height;

	update_tile_info();
}

float Film::get_exposure() const
{
	return exposure;
}
void Film::get_size(uint32_t& width, uint32_t& height) const
{
	width = (this)->width;
	height = (this)->height;
}
void Film::get_tile_size(uint32_t& width, uint32_t& height) const
{
	width = (this)->tile_width;
	height = (this)->tile_height;
}

GLuint Film::get_image_tex() const
{
	return image_tex;
}
uint32_t Film::get_id() const
{
	return id;
}
dFilm* Film::get_dptr() const
{
	return dptr;
}

void Film::clear()
{
	clear_dfilm(dptr);

	tile_id = 0;

	tile_x_pos = 0;
	tile_y_pos = 0;

	dptr->tile_x_pos = tile_x_pos;
	dptr->tile_y_pos = tile_y_pos;
}

void Film::copy_texture_to_film(const GLuint src_tex)
{
	glCopyImageSubData(src_tex, GL_TEXTURE_2D, 0, 0, 0, 0, image_tex, GL_TEXTURE_2D, 0, 0, 0, 0, width, height, 1);
}

void Film::update_tile_position()
{
	tile_id = (tile_id + 1) % nmb_tiles;

	tile_x_pos = tile_id % nmb_tile_cols;
	tile_y_pos = tile_id / nmb_tile_cols;

	dptr->tile_x_pos = tile_x_pos;
	dptr->tile_y_pos = tile_y_pos;
}

void Film::init_film_tex()
{
	glGenTextures(1, &image_tex);
	glBindTexture(GL_TEXTURE_2D, image_tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	glBindTexture(GL_TEXTURE_2D, 0);
}
void Film::init_dptr()
{
	checkCudaErrors(cudaMallocManaged(&dptr, sizeof(dFilm)));

	(this)->dptr->gamma = 1.f;
	(this)->dptr->exposure_time = 1.f;
	(this)->dptr->width = width;
	(this)->dptr->height = height;
	(this)->dptr->nmb_samples = nmb_samples;
	(this)->dptr->max_path_length = max_path_length;

	uint32_t nmb_pixels = width * height;

	/*
	checkCudaErrors(cudaMallocManaged(&dptr->paths, sizeof(Paths)));
	checkCudaErrors(cudaMalloc(&dptr->paths->throughput, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->length, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_ray, sizeof(dRay) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_isect, sizeof(Isect) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_pdf, sizeof(float) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_f, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->f, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_cosine, sizeof(float) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_ray, sizeof(dRay) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_id, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_visible, sizeof(bool) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->pixel_id, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->uv, sizeof(jek::Vec2f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->dead, sizeof(bool) * nmb_pixels));
	*/
	checkCudaErrors(cudaMallocManaged(&dptr->paths, sizeof(Paths)));
	checkCudaErrors(cudaMalloc(&dptr->paths->f_light, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->f_brdf, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->f_sample, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->Li_light, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->Li_brdf, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->beta, sizeof(jek::Vec3f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_light, sizeof(jek::Vec2f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_brdf, sizeof(jek::Vec2f) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_sample, sizeof(float) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ray, sizeof(dRay) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->ray_light, sizeof(dRay) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->len, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_id, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->isect, sizeof(Isect) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->dead, sizeof(bool) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->paths->visible, sizeof(bool) * nmb_pixels));

	checkCudaErrors(cudaMalloc(&dptr->samples, sizeof(uint32_t) * nmb_pixels));
	checkCudaErrors(cudaMalloc(&dptr->Ld, sizeof(jek::Vec3f) * nmb_pixels));

	clear_dfilm(dptr);
}

void Film::update_tile_info()
{
	nmb_tile_cols = ceil((float)width / (float)tile_width);
	nmb_tile_rows = ceil((float)height / (float)tile_height);

	nmb_tiles = nmb_tile_cols * nmb_tile_rows;
}

void Film::update_tex_size()
{
	glBindTexture(GL_TEXTURE_2D, image_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
}
void Film::update_dptr_size()
{
	(this)->dptr->width = width;
	(this)->dptr->height = height;
}

void Film::update_path_size()
{
	/*
	checkCudaErrors(cudaFree(dptr->paths->throughput));
	checkCudaErrors(cudaFree(dptr->paths->length));
	checkCudaErrors(cudaFree(dptr->paths->ext_ray));
	checkCudaErrors(cudaFree(dptr->paths->ext_isect));
	checkCudaErrors(cudaFree(dptr->paths->ext_pdf));
	checkCudaErrors(cudaFree(dptr->paths->ext_f));
	checkCudaErrors(cudaFree(dptr->paths->f));
	checkCudaErrors(cudaFree(dptr->paths->ext_cosine));
	checkCudaErrors(cudaFree(dptr->paths->light_ray));
	checkCudaErrors(cudaFree(dptr->paths->light_id));
	checkCudaErrors(cudaFree(dptr->paths->light_visible));
	checkCudaErrors(cudaFree(dptr->paths->pixel_id));
	checkCudaErrors(cudaFree(dptr->paths->uv));
	checkCudaErrors(cudaFree(dptr->paths->dead));

	checkCudaErrors(cudaFree(dptr->samples));
	checkCudaErrors(cudaFree(dptr->radiance));

	pathpool_size = width * height;
	checkCudaErrors(cudaMalloc(&dptr->paths->throughput, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->length, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_ray, sizeof(dRay) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_isect, sizeof(Isect) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_pdf, sizeof(float) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_f, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->f, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ext_cosine, sizeof(float) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_ray, sizeof(dRay) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_id, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_visible, sizeof(bool) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->pixel_id, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->uv, sizeof(jek::Vec2f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->dead, sizeof(bool) * pathpool_size));

	checkCudaErrors(cudaMalloc(&dptr->samples, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->radiance, sizeof(jek::Vec3f) * pathpool_size));
	*/

	checkCudaErrors(cudaFree(dptr->paths->f_light));
	checkCudaErrors(cudaFree(dptr->paths->f_brdf));
	checkCudaErrors(cudaFree(dptr->paths->f_sample));
	checkCudaErrors(cudaFree(dptr->paths->Li_light));
	checkCudaErrors(cudaFree(dptr->paths->Li_brdf));
	checkCudaErrors(cudaFree(dptr->paths->beta));
	checkCudaErrors(cudaFree(dptr->paths->pdf_light));
	checkCudaErrors(cudaFree(dptr->paths->pdf_brdf));
	checkCudaErrors(cudaFree(dptr->paths->pdf_sample));
	checkCudaErrors(cudaFree(dptr->paths->ray));
	checkCudaErrors(cudaFree(dptr->paths->ray_light));
	checkCudaErrors(cudaFree(dptr->paths->len));
	checkCudaErrors(cudaFree(dptr->paths->light_id));
	checkCudaErrors(cudaFree(dptr->paths->isect));
	checkCudaErrors(cudaFree(dptr->paths->dead));
	checkCudaErrors(cudaFree(dptr->paths->visible));

	checkCudaErrors(cudaFree(dptr->samples));
	checkCudaErrors(cudaFree(dptr->Ld));

	pathpool_size = width * height;
	checkCudaErrors(cudaMalloc(&dptr->paths->f_light, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->f_brdf, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->f_sample, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->Li_light, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->Li_brdf, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->beta, sizeof(jek::Vec3f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_light, sizeof(jek::Vec2f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_brdf, sizeof(jek::Vec2f) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->pdf_sample, sizeof(float) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ray, sizeof(dRay) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->ray_light, sizeof(dRay) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->len, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->light_id, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->isect, sizeof(Isect) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->dead, sizeof(bool) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->paths->visible, sizeof(bool) * pathpool_size));

	checkCudaErrors(cudaMalloc(&dptr->samples, sizeof(uint32_t) * pathpool_size));
	checkCudaErrors(cudaMalloc(&dptr->Ld, sizeof(jek::Vec3f) * pathpool_size));
}

void Film::update(const std::string& msg)
{
	clear();
}
