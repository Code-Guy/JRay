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

#include "linear_math.h"
#include <string>

#define FW_F32_MIN          (1.175494351e-38f)
#define FW_F32_MAX          (3.402823466e+38f)
#define NULL 0
#define FW_ASSERT(X) ((void)0)  
#define FLOAT_EQUAL_EPSILON 0.0001f

typedef unsigned char U8;
typedef unsigned short U16;
typedef unsigned int U32;
typedef unsigned long U64;
typedef signed char S8;
typedef signed short S16;
typedef signed int S32;
typedef signed long S64;
typedef float F32;
typedef double F64;

inline F32          bitsToFloat(U32 a)         { return *(F32*)&a; }
inline U32          floatToBits(F32 a)         { return *(U32*)&a; }

inline int max1i(const int& a, const int& b){ return (a < b) ? b : a; }
inline int min1i(const int& a, const int& b){ return (a > b) ? b : a; }
inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
inline float degree(float radian) { return radian * 180.0f / M_PI; }
inline float __host__ __device__ radian(float degree) { return degree / 180.0f * M_PI; }
inline float mod(float x, float y) { return x - y * floorf(x / y); }
inline unsigned int wangHash(unsigned int a)
{
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

inline __device__ Vec3f absmax3f(const Vec3f& v1, const Vec3f& v2) { return Vec3f(v1.x*v1.x > v2.x*v2.x ? v1.x : v2.x, v1.y*v1.y > v2.y*v2.y ? v1.y : v2.y, v1.z*v1.z > v2.z*v2.z ? v1.z : v2.z); }
inline __device__ unsigned char gammaCorrect(float c) { return unsigned char(std::powf(clampf(c, 0.0f, 1.0f), 0.45454545f) * 255); }
inline __device__ float3 toFloat3(const Vec3f &v) { return make_float3(v.x, v.y, v.z); };
inline __device__ Vec3f toVec3f(const float3 &v) { return Vec3f(v.x, v.y, v.z); };
inline __device__ Vec3f toVec3f(const float4 &v) { return Vec3f(v.x, v.y, v.z); };
inline __device__ Vec3f reflect(Vec3f i, Vec3f n) { return i - n * dot(n, i) * 2.0f; }

inline __device__ float3 min3(const float3& v1, const float3& v2) { return make_float3(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __device__ float3 max3(const float3& v1, const float3& v2) { return make_float3(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __device__ float min1(float3 v) { return v.x < v.y ? (v.x < v.z ? v.x : v.z) : (v.y < v.z ? v.y : v.z); }
inline __device__ float max1(float3 v) { return v.x > v.y ? (v.x > v.z ? v.x : v.z) : (v.y > v.z ? v.y : v.z); }
inline __device__ float min1(const Vec3f &v) { return v.x < v.y ? (v.x < v.z ? v.x : v.z) : (v.y < v.z ? v.y : v.z); }
inline __device__ float max1(const Vec3f &v) { return v.x > v.y ? (v.x > v.z ? v.x : v.z) : (v.y > v.z ? v.y : v.z); }

inline __device__ void swap2(int& a, int& b) { int temp = a; a = b; b = temp; }
inline __device__ bool floatEqual(float a, float b) { return fabsf(a - b) < FLOAT_EQUAL_EPSILON; }