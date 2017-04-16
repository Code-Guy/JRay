#include "sceneloader.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>


unsigned int verticesNum;
Vertex *vertices;
unsigned int trianglesNum;
Triangle *triangles;

bool loadObj(const char *fileName, Vec3f bottom, float size)
{
	Assimp::Importer importer;
	const aiScene *scene = importer.ReadFile(fileName, aiProcess_Triangulate | 
		aiProcess_JoinIdenticalVertices |
		aiProcess_GenNormals |
		aiProcess_CalcTangentSpace);

	if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		printf("ERROR::ASSIMP::%s\n", importer.GetErrorString());
		return false;
	}

	// only load first mesh now!
	const aiMesh *mesh = scene->mMeshes[0];
	verticesNum = mesh->mNumVertices;
	vertices = new Vertex[verticesNum];
	trianglesNum = mesh->mNumFaces;
	triangles = new Triangle[trianglesNum];

	Vec3f min = Vec3f(FLT_MAX, FLT_MAX, FLT_MAX);
	Vec3f max = Vec3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);

	for (int i = 0; i < verticesNum; i++)
	{
		vertices[i].position = Vec3f(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
		vertices[i].normal = Vec3f(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
		vertices[i].texCoord = mesh->mTextureCoords[0] ? 
			Vec2f(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : Vec2f(0.0f, 0.0f);
		vertices[i].tangent = Vec3f(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);
		vertices[i].bitangent = Vec3f(mesh->mBitangents[i].x, mesh->mBitangents[i].y, mesh->mBitangents[i].z);

		min = min3f(min, vertices[i].position);
		max = max3f(max, vertices[i].position);
	}

	// transform vertices
	Vec3f objCenter = (min + max) / 2;
	float objSize = (max - min).max();
	float scale = size / objSize;
	Vec3f offset = Vec3f(bottom.x, bottom.y + (max.y - min.y) * scale / 2.0f, bottom.z);

	for (int i = 0; i < verticesNum; i++)
	{
		vertices[i].position -= objCenter;
		vertices[i].position *= scale;
		vertices[i].position += offset;
	}

	for (int i = 0; i < trianglesNum; i++)
	{
		triangles[i].idx = Vec3i(mesh->mFaces[i].mIndices[0],
			mesh->mFaces[i].mIndices[1],
			mesh->mFaces[i].mIndices[2]);
	}

	return true;
}