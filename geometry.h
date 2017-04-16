#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "linear_math.h"

struct Vertex
{
	Vec3f position;
	Vec3f normal;
	Vec2f texCoord;
	Vec3f tangent;
	Vec3f bitangent;
};

struct Triangle
{
	Vec3i idx;
};

#endif // GEOMETRY_H
