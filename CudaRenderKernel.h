/*
*  Copyright (c) 2009-2011, NVIDIA Corporation
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of NVIDIA Corporation nor the
*        names of its contributors may be used to endorse or promote products
*        derived from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
*  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
*  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
*  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
*  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <cuda.h>
#include "linear_math.h"
#include "camera.h"

//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------
#define WIDTH 1280
#define HEIGHT 720
#define FOV 45
#define STACK_SIZE 64
#define DYNAMIC_FETCH_THRESHOLD 20
#define SPP 4
#define BOUNCE 6
#define RR_BOUNCE 3
#define F32_MIN 1.175494351e-38f
#define F32_MAX 3.402823466e+38f
#define HDR_WIDTH 3200
#define HDR_HEIGHT 1600
#define ENTRY_POINT_SENTINEL 0x76543210
#define MAX_BLOCK_HEIGHT 6

#define SPHERE_EPSILON 0.01f
#define BOX_EPSILON 0.01f
#define DIFFUSE_EPSILON 0.01f
#define METAL_EPSILON 0.01f
#define REFLECT_EPSILON 0.01f
#define TOTAL_INTERNAL_REFLECT_EPSILON 0.01f
#define REFRACT_REFLECT_EPSILON 0.01f
#define REFRACT_EPSILON 0.0f
#define PHONG_EXPONENT 20
#define M_PI2 6.283185307

//------------------------------------------------------------------------
// Interfaces.
//------------------------------------------------------------------------
void initTextures(float4 *gpuVertexIndices, float4 *gpuNormals, float4 *gpuTangents, float4 *gpuBiangents, float2 *gpuTexCoords, int trianglesNum, int vertexNum,
	float4 *gpuTriWoops, S32 *gpuTriIndices, cudaArray *gpuHDREnv, int triIndicesSize,
	cudaArray *gpuDiffuseImage, cudaArray *gpuNormalImage);

void cudaRender(Vec3f *frameBuffer, Vec3f *accumBuffer, const float4 *gpuNodes,
	Camera *camera, unsigned int frameNum, unsigned int hashedFrameNum);

//------------------------------------------------------------------------
// BVH memory layout.
//------------------------------------------------------------------------

enum BVHLayout
{
	BVHLayout_AOS_AOS = 0,              // Nodes = array-of-structures, triangles = array-of-structures. Used by tesla_xxx kernels.
	BVHLayout_AOS_SOA,                  // Nodes = array-of-structures, triangles = structure-of-arrays.
	BVHLayout_SOA_AOS,                  // Nodes = structure-of-arrays, triangles = array-of-structures.
	BVHLayout_SOA_SOA,                  // Nodes = structure-of-arrays, triangles = structure-of-arrays.
	BVHLayout_Compact,                  // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
	BVHLayout_Compact2,                 // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.

	BVHLayout_Max
};
