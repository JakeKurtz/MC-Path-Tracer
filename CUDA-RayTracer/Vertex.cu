#include "Vertex.h"

Vertex::operator dVertex() const
{
	dVertex v = dVertex();
	v.position = jek::Vec3f(position.x, position.y, position.z);
	v.normal = jek::Vec3f(normal.x, normal.y, normal.z);
	v.texcoords = jek::Vec2f(texCoords.x, texCoords.y);
	v.tangent = jek::Vec3f(tangent.x, tangent.y, tangent.z);
	v.bitangent = jek::Vec3f(bitangent.x, bitangent.y, bitangent.z);
	return v;
}
Vertex& Vertex::operator= (const dVertex& v)
{
	position = v.position;
	normal = v.normal;
	texCoords = v.texcoords;
	tangent = v.tangent;
	bitangent = v.bitangent;
	return *this;
}