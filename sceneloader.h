#ifndef SCENELOADER_H
#define SCENELOADER_H

#include "geometry.h"

extern unsigned int verticesNum;
extern Vertex *vertices;
extern unsigned int trianglesNum;
extern Triangle *triangles;

bool loadObj(const char *fileName, Vec3f bottom = Vec3f(0.0f, 0.0f, 0.0f), float size = 1.0f);

#endif // SCENELOADER_H
