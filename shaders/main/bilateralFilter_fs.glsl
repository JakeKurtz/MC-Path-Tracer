#version 330 core

out vec4 FragColor;

in vec2 TexCoords;

#define SIGMA 10.0
#define BSIGMA 0.1
#define MSIZE 15

uniform sampler2D image;
  
uniform bool horizontal;
uniform vec3 scale;
uniform float r;

float normpdf(float x, float sigma)
{
    return 0.39894 * exp(-0.5 * x * x / (sigma * sigma)) / sigma;
}

float normpdf3(vec3 v, float sigma)
{
    return 0.39894 * exp(-0.5 * dot(v, v) / (sigma * sigma)) / sigma;
}

void main()
{
	vec3 c = texture(image, TexCoords).rgb;
	vec2 tex_size = textureSize(image, 0).xy;
	//declare stuff
	const int kSize = (MSIZE - 1) / 2;
	float kernel[MSIZE];
	vec3 final_colour = vec3(0.0);

	//create the 1-D kernel
	float Z = 0.0;
	for (int j = 0; j <= kSize; ++j)
	{
		kernel[kSize + j] = kernel[kSize - j] = normpdf(float(j), SIGMA);
	}

	vec3 cc;
	float factor;
	float bZ = 1.0 / normpdf(0.0, BSIGMA);
	//read out the texels
	for (int i = -kSize; i <= kSize; ++i)
	{
		for (int j = -kSize; j <= kSize; ++j)
		{
			vec2 offset = vec2(i, j) / tex_size;
			cc = texture(image, TexCoords + offset).rgb;
			factor = normpdf3(cc - c, BSIGMA) * bZ * kernel[kSize + j] * kernel[kSize + i];
			Z += factor;
			final_colour += factor * cc;
		}
	}

	FragColor = vec4(final_colour / Z, 1.0);
}