#include "CudaRenderKernel.h"
#include "helper_math.h"
#include "cutil.cuh"
#include <curand_kernel.h>
#include <device_launch_parameters.h>

enum class Material { Diffuse, Reflective, Refractive, Metal };
enum class GeomType { SPHERE, BOX, TRIANGLE, NONE };

texture<float4, 1, cudaReadModeElementType> vertexIndicesTexture;
texture<float4, 1, cudaReadModeElementType> normalsTexture;
texture<float2, 1, cudaReadModeElementType> texCoordsTexture;
texture<float4, 1, cudaReadModeElementType> tangentsTexture;
texture<float4, 1, cudaReadModeElementType> bitangentsTexture;
texture<float4, 1, cudaReadModeElementType> triWoopsTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;
texture<float4, 2, cudaReadModeElementType> hdrTexture;
texture<float4, 2, cudaReadModeElementType> diffuseTexture;
texture<float4, 2, cudaReadModeElementType> normalTexture;

struct Intersection
{
	GeomType geomType = GeomType::NONE;
	float t = FLT_MAX;
	int idx = -1;
	float u, v;
};

union Color
{
	float data;
	uchar4 components;
};

struct Ray
{
	__device__ Ray(float3 pos, float3 dir) :
		pos(pos), dir(dir) {}

	float3 pos;
	float3 dir;
};

struct Sphere
{
	float radius;
	float3 pos, emissionColor, mainColor;
	Material material;

	__device__ float intersect(const Ray &ray) const
	{
		float t;
		float3 dis = pos - ray.pos;
		float proj = dot(dis, ray.dir);
		float delta = proj * proj - dot(dis, dis) + radius * radius;

		if (delta < 0) return 0;

		delta = sqrtf(delta);
		return (t = proj - delta) > SPHERE_EPSILON ? t : ((t = proj + delta) > SPHERE_EPSILON ? t : 0);
	}
};

struct Box
{
	float3 min;
	float3 max;
	float3 emissionColor;
	float3 mainColor;
	Material material;

	__device__ float intersect(const Ray &ray) const
	{
		float3 tmin = (min - ray.pos) / ray.dir;
		float3 tmax = (max - ray.pos) / ray.dir;
		float3 rmin = min3(tmin, tmax);
		float3 rmax = max3(tmin, tmax);
		float minmax = min1(rmax);
		float maxmin = max1(rmin);

		if (minmax >= maxmin)
			return maxmin > BOX_EPSILON ? maxmin : 0;
		return 0;
	}

	__device__ Vec3f normalAt(const Vec3f &p) const
	{
		if (fabs(min.x - p.x) < BOX_EPSILON) return Vec3f(-1.0f, 0.0f, 0.0f);
		if (fabs(max.x - p.x) < BOX_EPSILON) return Vec3f(1.0f, 0.0f, 0.0f);
		if (fabs(min.y - p.y) < BOX_EPSILON) return Vec3f(0.0f, -1.0f, 0.0f);
		if (fabs(max.y - p.y) < BOX_EPSILON) return Vec3f(0.0f, 1.0f, 0.0f);
		if (fabs(min.z - p.z) < BOX_EPSILON) return Vec3f(0.0f, 0.0f, -1.0f);
		if (fabs(max.z - p.z) < BOX_EPSILON) return Vec3f(0.0f, 0.0f, 1.0f);

		return Vec3f(0.0f, 0.0f, 0.0f);
	}
};

__constant__ Sphere spheres[] =
{
	{ 20.0f,{ 30.0f, 20.0f, 30.0f },{ 0.0f, 0.0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, Material::Reflective }, // sphere 1
};

__constant__ Box boxes[] =
{
	{ { -100.0f, -1.0f, -100.0f },{ 100.0f, 0.0f, 100.0f },{ 0.0f, 0.0f, 0.0f },{ 0.75f, 0.75f, 0.75f }, Material::Diffuse }, // ground
	{ { -40.0f, 0.0f, -40.0f },{ -10.0f, 50.0f, -10.0f },{ 0.0f, 0.0f, 0.0f },{ 0.5f, 1.0f, 0.5f }, Material::Diffuse } // box 1
};

__device__ void intersectsSpheres(const float3 &rayPos, const float3 &rayDir, Intersection &intersection)
{
	int sphereNum = sizeof(spheres) / sizeof(Sphere);
	for (int i = 0; i < sphereNum; i++)
	{
		float t = spheres[i].intersect(Ray(rayPos, rayDir));
		if (t != 0 && t < intersection.t)
		{
			intersection.geomType = GeomType::SPHERE;
			intersection.t = t;
			intersection.idx = i;
		}
	}
}

__device__ void intersectsBoxes(const float3 &rayPos, const float3 &rayDir, Intersection &intersection)
{
	int boxNum = sizeof(boxes) / sizeof(Box);
	for (int i = 0; i < boxNum; i++)
	{
		float t = boxes[i].intersect(Ray(rayPos, rayDir));
		if (t != 0 && t < intersection.t)
		{
			intersection.geomType = GeomType::BOX;
			intersection.t = t;
			intersection.idx = i;
		}
	}
}

__device__ void intersectsBVH(const float3 &rayPos, const float3 &rayDir, const float4 *gpuNodes, Vec3f &triNormal, Intersection &intersection)
{
	int stack[STACK_SIZE];
	stack[0] = ENTRY_POINT_SENTINEL;

	float posx = rayPos.x, posy = rayPos.y, posz = rayPos.z;
	float dirx = rayDir.x, diry = rayDir.y, dirz = rayDir.z;
	float ooeps = exp2f(-80.0f);
	float idirx = 1.0f / (fabsf(dirx) > ooeps ? dirx : copysignf(ooeps, dirx));
	float idiry = 1.0f / (fabsf(diry) > ooeps ? diry : copysignf(ooeps, diry));
	float idirz = 1.0f / (fabsf(dirz) > ooeps ? dirz : copysignf(ooeps, dirz));
	float oodx = posx * idirx, oody = posy * idiry, oodz = posz * idirz;
	float tmin = 0.01f, ht = intersection.t;
	float hu = 0.0f, hv = 0.0f;
	int hidx = -1;
	char *stackAddr = (char *)&stack[0];
	int leafAddr = 0;
	int nodeAddr = 0;

	while (nodeAddr != ENTRY_POINT_SENTINEL)
	{
		while (nodeAddr >= 0 && nodeAddr != ENTRY_POINT_SENTINEL)
		{
			float4 *node = (float4 *)((char *)gpuNodes + nodeAddr);
			float4 n0xy = node[0];
			float4 n1xy = node[1];
			float4 nz = node[2];

			// ray-box intersection
			float c0lox = n0xy.x * idirx - oodx;
			float c0hix = n0xy.y * idirx - oodx;
			float c0loy = n0xy.z * idiry - oody;
			float c0hiy = n0xy.w * idiry - oody;
			float c0loz = nz.x   * idirz - oodz;
			float c0hiz = nz.y   * idirz - oodz;
			float c1loz = nz.z   * idirz - oodz;
			float c1hiz = nz.w   * idirz - oodz;
			float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
			float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, ht);
			float c1lox = n1xy.x * idirx - oodx;
			float c1hix = n1xy.y * idirx - oodx;
			float c1loy = n1xy.z * idiry - oody;
			float c1hiy = n1xy.w * idiry - oody;
			float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, ht);
			bool traverseChild0 = c0min <= c0max;
			bool traverseChild1 = c1min <= c1max;

			if (!traverseChild0 && !traverseChild1)
			{
				nodeAddr = *(int*)stackAddr;
				stackAddr -= 4;
			}
			else
			{
				int2 cnodes = *(int2*)&node[3];
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

				if (traverseChild0 && traverseChild1)
				{
					if (c1min < c0min)
						swap2(nodeAddr, cnodes.y);
					stackAddr += 4;
					*(int*)stackAddr = cnodes.y;
				}
			}

			if (nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				nodeAddr = *(int*)stackAddr;
				stackAddr -= 4;
			}

			unsigned int mask;
			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.ballot.b32    %0,p;       \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));
			if (!mask)
				break;
		}

		while (leafAddr < 0)
		{
			for (int triAddr = ~leafAddr;; triAddr += 3)
			{
				float4 v00 = tex1Dfetch(triWoopsTexture, triAddr);
				if (__float_as_int(v00.x) == 0x80000000)
					break;
				float Oz = v00.w - posx*v00.x - posy*v00.y - posz*v00.z;
				float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
				float t = Oz * invDz;

				if (t > tmin && t < ht)
				{
					float4 v11 = tex1Dfetch(triWoopsTexture, triAddr + 1);
					float Ox = v11.w + posx*v11.x + posy*v11.y + posz*v11.z;
					float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;
					float u = Ox + t * Dx;

					if (u >= 0.0f && u <= 1.0f)
					{
						float4 v22 = tex1Dfetch(triWoopsTexture, triAddr + 2);
						float Oy = v22.w + posx*v22.x + posy*v22.y + posz*v22.z;
						float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
						float v = Oy + t*Dy;

						if (v >= 0.0f && u + v <= 1.0f)
						{
							ht = t;
							hu = u;
							hv = v;
							hidx = triAddr;
							triNormal = cross(Vec3f(v22.x, v22.y, v22.z), Vec3f(v11.x, v11.y, v11.z));
						}
					}
				}
			}
			leafAddr = nodeAddr;
			if (nodeAddr < 0)
			{
				nodeAddr = *(int*)stackAddr;
				stackAddr -= 4;
			}
		}
	}

	if (hidx != -1)
	{
		hidx = tex1Dfetch(triIndicesTexture, hidx);
		intersection.geomType = GeomType::TRIANGLE;
		intersection.idx = hidx;
		intersection.t = ht;
		intersection.u = hu;
		intersection.v = hv;
	}
}

__device__ Vec3f sampleHDR(const Vec3f &rayDir)
{
	float theta = atan2f(rayDir.x, rayDir.z);
	if (theta < 0) theta += M_PI2;
	float phi = acosf(rayDir.y);

	float u = theta / M_PI2;
	float v = phi / M_PI;

	float4 color = tex2D(hdrTexture, u, v);
	return Vec3f(color.x, color.y, color.z);
}

__device__ Vec3f sampleDiffuseTexture(const Intersection &intersection)
{
	float4 vidx = tex1Dfetch(vertexIndicesTexture, intersection.idx);

	float2 uv0 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.x));
	float2 uv1 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.y));
	float2 uv2 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.z));

	float2 uv = uv0 * intersection.u +
		uv1 * intersection.v + uv2 * (1 - intersection.u - intersection.v);

	float4 color = tex2D(diffuseTexture, uv.x, 1 - uv.y);
	return Vec3f(color.x, color.y, color.z);
}

__device__ Vec3f sampleProceduralTexture(const Intersection &intersection)
{
	float4 vidx = tex1Dfetch(vertexIndicesTexture, intersection.idx);

	float2 uv0 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.x));
	float2 uv1 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.y));
	float2 uv2 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.z));

	float u = uv0.x * intersection.u +
		uv1.x * intersection.v + uv2.x * (1 - intersection.u - intersection.v);
	float v = uv0.y * intersection.u +
		uv1.y * intersection.v + uv2.y * (1 - intersection.u - intersection.v);

	//float val = noise(10 * u, 10 * v, 0.8);
	float val = 100 * noise(u, v, 0.8);
	val -= floor(val);
	//float val = u * u + v * v;
	return Vec3f(val, val, val) * Vec3f(255, 193,  38) / Vec3f(255, 255, 255);
}

__device__ Vec3f getSmoothNormal(const Intersection &intersection)
{
	float4 vidx = tex1Dfetch(vertexIndicesTexture, intersection.idx);

	float4 n0 = tex1Dfetch(normalsTexture, __float_as_int(vidx.x));
	float4 n1 = tex1Dfetch(normalsTexture, __float_as_int(vidx.y));
	float4 n2 = tex1Dfetch(normalsTexture, __float_as_int(vidx.z));

	//float4 t0 = tex1Dfetch(tangentsTexture, __float_as_int(vidx.x));
	//float4 t1 = tex1Dfetch(tangentsTexture, __float_as_int(vidx.y));
	//float4 t2 = tex1Dfetch(tangentsTexture, __float_as_int(vidx.z));

	//float4 b0 = tex1Dfetch(bitangentsTexture, __float_as_int(vidx.x));
	//float4 b1 = tex1Dfetch(bitangentsTexture, __float_as_int(vidx.y));
	//float4 b2 = tex1Dfetch(bitangentsTexture, __float_as_int(vidx.z));

	Vec3f n = toVec3f(n0) * intersection.u + 
		toVec3f(n1) * intersection.v + toVec3f(n2) * (1 - intersection.u - intersection.v);
	//Vec3f t = toVec3f(t0) * intersection.u +
	//	toVec3f(t1) * intersection.v + toVec3f(t2) * (1 - intersection.u - intersection.v);
	//Vec3f b = toVec3f(b0) * intersection.u +
	//	toVec3f(b1) * intersection.v + toVec3f(b2) * (1 - intersection.u - intersection.v);

	//float2 uv0 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.x));
	//float2 uv1 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.y));
	//float2 uv2 = tex1Dfetch(texCoordsTexture, __float_as_int(vidx.z));

	//float2 uv = uv0 * intersection.u +
	//	uv1 * intersection.v + uv2 * (1 - intersection.u - intersection.v);

	//float4 texNormal = tex2D(normalTexture, uv.x, (1 - uv.y));
	//texNormal = normalize(texNormal * 2.0f - 1.0f);

	//n.normalize(); // y axis
	//t.normalize(); // x axis
	//b.normalize(); // z axis

	//Mat3 tnb(t, n, b);
	//
	//Vec3f normal = tnb * toVec3f(texNormal); normal.normalize();
	return n;
}

__device__ Vec3f pathTrace(Vec3f &rayPos, Vec3f &rayDir, const float4 *gpuNodes, curandState *randstate)
{
	Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f);
	Vec3f accumColor = Vec3f(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < BOUNCE; i++)
	{
		Intersection intersection;
		Vec3f p;
		Vec3f n, nl;
		Vec3f emissionColor, mainColor;
		Vec3f nextRayPos;
		Vec3f nextRayDir;
		Material material;

		float3 rayPosf3 = toFloat3(rayPos);
		float3 rayDirf3 = toFloat3(rayDir);

		Vec3f triNormal;
		intersectsSpheres(rayPosf3, rayDirf3, intersection);
		intersectsBoxes(rayPosf3, rayDirf3, intersection);
		intersectsBVH(rayPosf3, rayDirf3, gpuNodes, triNormal, intersection);

		if (!(intersection.t < FLT_MAX))
		{
			Vec3f hdrColor = sampleHDR(rayDir);
			//Vec3f hdrColor = Vec3f(0.7f, 0.7f, 0.7f);
			accumColor += mask * hdrColor;
			return accumColor;
		}

		p = rayPos + rayDir * intersection.t;
		if (intersection.geomType == GeomType::SPHERE)
		{
			Sphere &sphere = spheres[intersection.idx];
			emissionColor = toVec3f(sphere.emissionColor);
			mainColor = toVec3f(sphere.mainColor);
			n = p - toVec3f(sphere.pos); n.normalize();
			material = sphere.material;
		}
		else if (intersection.geomType == GeomType::BOX)
		{
			Box &box = boxes[intersection.idx];
			emissionColor = toVec3f(box.emissionColor);
			mainColor = toVec3f(box.mainColor);
			n = box.normalAt(p);
			material = box.material;
		}
		else if (intersection.geomType == GeomType::TRIANGLE)
		{
			emissionColor = Vec3f(0.0f, 0.0f, 0.0f);
			mainColor = Vec3f(1.0f, 1.0f, 1.0f);
			//mainColor = sampleDiffuseTexture(intersection);
			//mainColor = sampleProceduralTexture(intersection);
			n = getSmoothNormal(intersection);
			material = Material::Diffuse;
		}

		nl = dot(n, rayDir) < 0 ? n : n * -1;
		accumColor += mask * emissionColor;

		if (material == Material::Diffuse)
		{
			float phi = curand_uniform(randstate) * M_PI * 2;
			float r2 = curand_uniform(randstate);
			float r2s = std::sqrtf(r2);

			Vec3f w = nl;
			Vec3f u = cross((fabs(w.x) > fabs(w.y) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
			Vec3f v = cross(w, u);

			nextRayPos = p + nl * DIFFUSE_EPSILON;
			nextRayDir = u*cos(phi)*r2s + v*sin(phi)*r2s + w*sqrtf(1 - r2); nextRayDir.normalize();

			mask *= mainColor * dot(nextRayDir, nl) * 2;
		}
		else if (material == Material::Metal)
		{
			float phi = curand_uniform(randstate) * M_PI * 2;
			float r2 = curand_uniform(randstate);
			float cosTheta = powf(1 - r2, 1.0f / (PHONG_EXPONENT + 1));
			float sinTheta = sqrtf(1 - cosTheta * cosTheta);

			Vec3f w = reflect(rayDir, n);
			Vec3f u = cross((fabs(w.x) > fabs(w.y) ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)), w); u.normalize();
			Vec3f v = cross(w, u);

			nextRayPos = p + nl * METAL_EPSILON;
			nextRayDir = u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta; nextRayDir.normalize();

			mask *= mainColor;
		}
		else if (material == Material::Reflective)
		{
			nextRayDir = reflect(rayDir, n);
			nextRayPos = p + nl * REFLECT_EPSILON;
			mask *= mainColor;
		}
		else if (material == Material::Refractive)
		{
			bool into = dot(n, nl) > 0;
			float nc = 1.0f;
			float nt = 1.5f;
			float nnt = into ? nc / nt : nt / nc;
			float ddn = dot(rayDir, nl);
			float cos2t = 1.0f - nnt*nnt * (1.0f - ddn*ddn);

			if (cos2t < 0.0f)
			{
				nextRayPos = p + nl * TOTAL_INTERNAL_REFLECT_EPSILON;
				nextRayDir = reflect(rayDir, n);
				mask *= mainColor;
			}
			else
			{
				Vec3f tdir = rayDir * nnt - n * ((into ? 1 : -1) * (ddn*nnt + sqrtf(cos2t))); tdir.normalize();
				float R0 = (nt - nc)*(nt - nc) / ((nt + nc)*(nt + nc));
				float c = 1.0f - (into ? -ddn : dot(tdir, n));
				float Re = R0 + (1.0f - R0) * c * c * c * c * c;
				float Tr = 1 - Re;
				float P = 0.25f + 0.5f * Re;
				float RP = Re / P;
				float TP = Tr / (1 - P);

				if (curand_uniform(randstate) < P)
				{
					nextRayPos = p + nl * REFRACT_REFLECT_EPSILON;
					nextRayDir = reflect(rayDir, n);
					mask *= mainColor * RP;
				}
				else
				{
					nextRayPos = p + nl * REFRACT_EPSILON;
					nextRayDir = tdir;
					mask *= mainColor * TP;
				}
			}
		}

		rayPos = nextRayPos;
		rayDir = nextRayDir;

		// Russian Roulette
		if (i > RR_BOUNCE)
		{
			float p = max1(accumColor);
			if (curand_uniform(randstate) > p) break;
			accumColor /= p;
		}
	}

	return accumColor;
}

__global__ void pathTraceKernel(Vec3f *frameBuffer, Vec3f *accumBuffer, const float4 *gpuNodes,
	Camera *camera, unsigned int frameNum, unsigned int hashedFrameNum)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) +
		(threadIdx.y * blockDim.x) + threadIdx.x;

	curandState randState;
	curand_init(hashedFrameNum + threadId, 0, 0, &randState);

	Vec3f pos = camera->pos;

	int i = (HEIGHT - y - 1) * WIDTH + x;
	int px = x;
	int py = HEIGHT - y - 1;

	Vec3f pixel = Vec3f(0.0f, 0.0f, 0.0f);

	for (int i = 0; i < SPP; i++)
	{
		Vec3f view = camera->view;
		Vec3f up = camera->up;
		Vec3f horizontalAxis = cross(view, up); horizontalAxis.normalize();
		Vec3f verticalAxis = cross(horizontalAxis, view); verticalAxis.normalize();

		Vec3f center = pos + view;
		Vec3f horizontal = horizontalAxis * tanf(radian(camera->fov.x / 2.0f));
		Vec3f vertical = verticalAxis * tanf(-radian(camera->fov.y / 2.0f));

		float jx = curand_uniform(&randState) - 0.5f;
		float jy = curand_uniform(&randState) - 0.5f;
		float sx = (jx + px) / (camera->res.x - 1);
		float sy = (jy + py) / (camera->res.y - 1);

		Vec3f pointOnPlaneOneUnitAwayFromEye = center + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
		Vec3f pointOnImagePlane = pos + ((pointOnPlaneOneUnitAwayFromEye - pos) * camera->focalDist);
		Vec3f aperturePoint;

		if (camera->aperRad != 0)
		{
			float r1 = curand_uniform(&randState);
			float r2 = curand_uniform(&randState);
			float angle = M_PI * r1 * 2.0f;
			float distance = camera->aperRad * sqrtf(r2);
			float ax = cos(angle) * distance;
			float ay = sin(angle) * distance;
			aperturePoint = pos + (horizontalAxis * ax) + (verticalAxis * ay);
		}
		else
		{
			aperturePoint = camera->pos;
		}

		Vec3f rayPos = aperturePoint;
		Vec3f rayDir = pointOnImagePlane - aperturePoint; rayDir.normalize();

		pixel += pathTrace(rayPos, rayDir, gpuNodes, &randState) / SPP;
	}

	accumBuffer[i] += pixel;
	Vec3f temp = accumBuffer[i] / frameNum;

	Color color;
	color.components = make_uchar4(gammaCorrect(temp.x), gammaCorrect(temp.y), gammaCorrect(temp.z), 255);
	frameBuffer[i] = Vec3f(x, y, color.data);
}

void initTextures(float4 *gpuVertexIndices, float4 *gpuNormals, float4 *gpuTangents, float4 *gpuBiangents, float2 *gpuTexCoords, int triangleNum, int vertexNum,
	float4 *gpuTriWoops, S32 *gpuTriIndices, cudaArray *gpuHDREnv, int triIndicesSize,
	cudaArray *gpuDiffuseImage, cudaArray *gpuNormalImage)
{
	cudaChannelFormatDesc channel0desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &vertexIndicesTexture, gpuVertexIndices, &channel0desc, triangleNum * sizeof(float4));

	cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &normalsTexture, gpuNormals, &channel1desc, vertexNum * sizeof(float4));

	cudaChannelFormatDesc channel7desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &tangentsTexture, gpuTangents, &channel7desc, vertexNum * sizeof(float4));

	cudaChannelFormatDesc channel8desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &bitangentsTexture, gpuBiangents, &channel8desc, vertexNum * sizeof(float4));

	cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float2>();
	cudaBindTexture(NULL, &texCoordsTexture, gpuTexCoords, &channel2desc, vertexNum * sizeof(float2));

	cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(NULL, &triWoopsTexture, gpuTriWoops, &channel3desc, triIndicesSize * sizeof(float4));

	cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<int>();
	cudaBindTexture(NULL, &triIndicesTexture, gpuTriIndices, &channel4desc, triIndicesSize * sizeof(int));

	hdrTexture.normalized = true;
	hdrTexture.addressMode[0] = cudaAddressModeWrap;
	hdrTexture.addressMode[1] = cudaAddressModeWrap;
	hdrTexture.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
	cudaBindTextureToArray(&hdrTexture, gpuHDREnv, &channel5desc);

	diffuseTexture.normalized = true;
	diffuseTexture.addressMode[0] = cudaAddressModeWrap;
	diffuseTexture.addressMode[1] = cudaAddressModeWrap;
	diffuseTexture.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc channel6desc = cudaCreateChannelDesc<float4>();
	cudaBindTextureToArray(&diffuseTexture, gpuDiffuseImage, &channel6desc);

	normalTexture.normalized = true;
	normalTexture.addressMode[0] = cudaAddressModeWrap;
	normalTexture.addressMode[1] = cudaAddressModeWrap;
	normalTexture.filterMode = cudaFilterModeLinear;
	cudaChannelFormatDesc channel9desc = cudaCreateChannelDesc<float4>();
	cudaBindTextureToArray(&normalTexture, gpuNormalImage, &channel9desc);
}

void cudaRender(Vec3f *frameBuffer, Vec3f *accumBuffer, const float4 *gpuNodes,
	Camera *camera, unsigned int frameNum, unsigned int hashedFrameNum)
{
	dim3 block(16, 16, 1);
	dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

	pathTraceKernel << <grid, block >> > (frameBuffer, accumBuffer, gpuNodes,
		camera, frameNum, hashedFrameNum);
}