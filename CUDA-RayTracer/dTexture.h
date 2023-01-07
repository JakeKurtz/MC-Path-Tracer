#pragma once
#include <string>
class dTexture
{
	unsigned int id;
	std::string name;
	std::string filepath;

	//GLenum target;
	//GLenum type;
	//GLenum format;
	//GLint internalformat;

	//GLint wrap;
	//GLint min_filter;
	//GLint mag_filter;

	int width;
	int height;
	int depth;
	int nrComponents;
};

