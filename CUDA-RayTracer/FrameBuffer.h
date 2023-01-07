#ifndef FBO_H
#define FBO_H

#include "GLCommon.h"

#include "Attachment.h"
#include "Texture.h"

#include <vector>

class FrameBuffer {
public:
	FrameBuffer(GLsizei width, GLsizei height);
	~FrameBuffer();

	void bind();
	void bind(GLsizei width, GLsizei height);

	void bind_rbo();
	void bind_rbo(GLsizei width, GLsizei height);

	void unbind();
	void unbind_rbo();

	int attach(GLenum attachmentType, GLint internalformat, GLenum format, GLenum type);
	int attach_rbo(GLenum renderbuffertarget, GLenum internalformat);

	unsigned int get_color_tex_id(int i = 0);
	unsigned int get_depth_tex_id();
	unsigned int get_stencil_tex_id();

	std::shared_ptr<Attachment> get_color_attachment(int i = 0);
	std::shared_ptr<Attachment> get_depth_attachment();
	std::shared_ptr<Attachment> get_stencil_attachment();

	int check_status();
	void construct();

	unsigned int get_id() { return id; };
	GLsizei get_width() { return width; };
	GLsizei get_height() { return height; };

	void set_attachment_size(GLsizei width, GLsizei height);

private:
	unsigned int id;
	unsigned int rbo_id;

	GLsizei width;
	GLsizei height;

	GLsizei attachment_width;
	GLsizei attachment_height;

	GLsizei rbo_width;
	GLsizei rbo_height;

	GLint rbo_internal_format;
	GLint rbo_target;

	std::shared_ptr<Attachment> depth_attachment = nullptr;
	std::shared_ptr<Attachment> stencil_attachment = nullptr;
	std::vector<std::shared_ptr<Attachment>> color_attachments;

	int glClearValue = 1;

	bool fbo_constructed = false;

	unsigned int nmb_color_attachemnts = 0;
	unsigned int nmb_depth_attachemnts = 0;
	unsigned int nmb_stencil_attachemnts = 0;

};
#endif