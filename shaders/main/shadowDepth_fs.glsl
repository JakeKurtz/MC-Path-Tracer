#version 330 core

out vec2 FragColor;

void main()
{             
	float depth = gl_FragCoord.z;

	float dx = dFdx(depth);
	float dy = dFdy(depth);
	float moment2 = depth * depth + 0.25 * (dx * dx + dy * dy);

	FragColor = vec2(depth, 0.f);
}