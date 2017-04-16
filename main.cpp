// c++ headers
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

// cuda headers
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "CudaRenderKernel.h"
#include "CudaBVH.h"
#include "HDRloader.h"
#include "camera.h"
#include "sceneloader.h"

GLuint vbo;
unsigned int frameNum = 0;
bool resetBuffer = false;
int nodesSize = 0;
int leafnodeCount = 0;
int triangleCount = 0;
int triWoopsSize = 0;
int triIndicesSize = 0;

// mouse event handlers
int lastX = 0, lastY = 0;
bool mouseLeftButtonPressed = false;
InteractiveCamera *interactiveCamera = nullptr;

// cpu data
Vec4i *cpuNodes = nullptr;
Vec4i *cpuTriWoops = nullptr;
S32 *cpuTriIndices = nullptr;
Vec4f *cpuHDREnv = nullptr;
Camera *cpuCamera = nullptr;
Vec4f *cpuNormals = nullptr;
Vec4f *cpuTangents = nullptr;
Vec4f *cpuBitangents = nullptr;
Vec2f *cpuTexCoords = nullptr;
Vec4i *cpuVertexIndices = nullptr;
Vec4f *cpuDiffuseImage = nullptr;
Vec4f *cpuNormalImage = nullptr;

// gpu data
float4 *gpuNodes = nullptr;
float4 *gpuTriWoops = nullptr;
S32 *gpuTriIndices = nullptr;
cudaArray *gpuHDREnv = nullptr;
Camera *gpuCamera = nullptr;
Vec3f *gpuAccumBuffer = nullptr;
Vec3f *gpuFrameBuffer = nullptr;
CudaBVH *gpuBVH = nullptr;
float4 *gpuNormals = nullptr;
float4 *gpuTangents = nullptr;
float4 *gpuBitangents = nullptr;
float2 *gpuTexCoords = nullptr;
float4 *gpuVertexIndices = nullptr;
cudaArray *gpuDiffuseImage = nullptr;
cudaArray *gpuNormalImage = nullptr;

void initCamera()
{
	if (interactiveCamera) delete interactiveCamera;
	if (!cpuCamera) cpuCamera = new Camera;

	interactiveCamera = new InteractiveCamera;
	interactiveCamera->setRes(WIDTH, HEIGHT);
	interactiveCamera->setFovx(FOV);
}

void initVBO()
{
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, WIDTH * HEIGHT * sizeof(Vec3f), 0, GL_DYNAMIC_DRAW);

	cudaGLRegisterBufferObject(vbo);
}

void initImage(const char *fileName, Vec4f *&cpuImage, cudaArray *&gpuImage)
{
	int width, height;
	int bpp;
	unsigned char *imageData = stbi_load(fileName, &width, &height, &bpp, 3);

	int imageSize = width * height * sizeof(float4);
	cpuImage = new Vec4f[width * height];
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			int idx = width * j + i;
			cpuImage[idx] = Vec4f(imageData[idx * 3] / 255.0f, imageData[idx * 3 + 1] / 255.0f, imageData[idx * 3 + 2] / 255.0f, 0.0f);
		}
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&gpuImage, &channelDesc, width, height);
	cudaMemcpyToArray(gpuImage, 0, 0, cpuImage, imageSize, cudaMemcpyHostToDevice);

	stbi_image_free(imageData);
	delete[] cpuImage;
}

bool initHDR()
{
	HDRImage hdrImage;
	if (HDRLoader::load("media/hdrs/Topanga_Forest_B_3k.hdr", hdrImage))
	{
		printf("HDR environment map loaded. Width: %d Height: %d\n", hdrImage.width, hdrImage.height);
	}
	else
	{
		printf("HDR environment map not found!\nAn HDR map is required as light source!\n");
		return false;
	}

	int hdrWidth = hdrImage.width;
	int hdrHeight = hdrImage.height;
	int hdrEnvSize = hdrWidth * hdrHeight * sizeof(float4);

	cpuHDREnv = new Vec4f[hdrWidth * hdrHeight];
	for (int i = 0; i < hdrWidth; i++)
	{
		for (int j = 0; j < hdrHeight; j++)
		{
			int idx = hdrWidth * j + i;
			cpuHDREnv[idx] = Vec4f(hdrImage.colors[idx * 3], hdrImage.colors[idx * 3 + 1], hdrImage.colors[idx * 3 + 2], 0.0f);
		}
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&gpuHDREnv, &channelDesc, hdrWidth, hdrHeight);
	cudaMemcpyToArray(gpuHDREnv, 0, 0, cpuHDREnv, hdrEnvSize, cudaMemcpyHostToDevice);
	delete[] cpuHDREnv;
}

void initBVH()
{
	loadObj("media/models/dragon.obj", Vec3f(-20.0f, 0.0f, 60.0f), 60.0f);
	Array<Scene::Triangle> tris;
	Array<Vec3f> verts;

	for (int i = 0; i < verticesNum; i++)
	{
		verts.add(vertices[i].position);
	}
	for (int i = 0; i < trianglesNum; i++)
	{
		Scene::Triangle tri;
		tri.vertices = triangles[i].idx;
		tris.add(tri);
	}

	Scene *scene = new Scene(trianglesNum, verticesNum, tris, verts);

	Platform defaultPlatform;
	BVH::BuildParams defaultParams;
	BVH::Stats stats;
	BVH bvh(scene, defaultPlatform, defaultParams);
	gpuBVH = new CudaBVH(bvh, BVHLayout_Compact);

	cpuNodes = gpuBVH->getGpuNodes();
	cpuTriWoops = gpuBVH->getGpuTriWoop();
	cpuTriIndices = gpuBVH->getGpuTriIndices();

	nodesSize = gpuBVH->getGpuNodesSize();
	triWoopsSize = gpuBVH->getGpuTriWoopSize();
	triIndicesSize = gpuBVH->getGpuTriIndicesSize();
	leafnodeCount = gpuBVH->getLeafnodeCount();
	triangleCount = gpuBVH->getTriCount();
}

void initCuda()
{
	cudaMalloc(&gpuAccumBuffer, WIDTH * HEIGHT * sizeof(Vec3f));
	cudaMalloc(&gpuCamera, sizeof(Camera));

	cudaMalloc(&gpuNodes, nodesSize * sizeof(float4));
	cudaMemcpy(gpuNodes, cpuNodes, nodesSize * sizeof(float4), cudaMemcpyHostToDevice);
	free(cpuNodes);

	cudaMalloc(&gpuTriWoops, triWoopsSize * sizeof(float4));
	cudaMemcpy(gpuTriWoops, cpuTriWoops, triWoopsSize * sizeof(float4), cudaMemcpyHostToDevice);
	free(cpuTriWoops);

	cudaMalloc(&gpuTriIndices, triIndicesSize * sizeof(S32));
	cudaMemcpy(gpuTriIndices, cpuTriIndices, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);
	free(cpuTriIndices);

	cpuNormals = new Vec4f[verticesNum];
	cpuTangents = new Vec4f[verticesNum];
	cpuBitangents = new Vec4f[verticesNum];
	cpuTexCoords = new Vec2f[verticesNum];
	for (int i = 0; i < verticesNum; i++)
	{
		cpuNormals[i] = Vec4f(vertices[i].normal, 0.0f);
		cpuTangents[i] = Vec4f(vertices[i].tangent, 0.0f);
		cpuBitangents[i] = Vec4f(vertices[i].bitangent, 0.0f);
		cpuTexCoords[i] = vertices[i].texCoord;
	}
	cudaMalloc(&gpuNormals, verticesNum * sizeof(Vec4f));
	cudaMemcpy(gpuNormals, cpuNormals, verticesNum * sizeof(Vec4f), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuTangents, verticesNum * sizeof(Vec4f));
	cudaMemcpy(gpuTangents, cpuTangents, verticesNum * sizeof(Vec4f), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuBitangents, verticesNum * sizeof(Vec4f));
	cudaMemcpy(gpuBitangents, cpuBitangents, verticesNum * sizeof(Vec4f), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuTexCoords, verticesNum * sizeof(Vec2f));
	cudaMemcpy(gpuTexCoords, cpuTexCoords, verticesNum * sizeof(Vec2f), cudaMemcpyHostToDevice);

	cpuVertexIndices = new Vec4i[trianglesNum];
	for (int i = 0; i < trianglesNum; i++)
	{
		cpuVertexIndices[i] = Vec4i(triangles[i].idx, 0);
	}
	cudaMalloc(&gpuVertexIndices, trianglesNum * sizeof(Vec4i));
	cudaMemcpy(gpuVertexIndices, cpuVertexIndices, trianglesNum * sizeof(Vec4i), cudaMemcpyHostToDevice);

	delete[] cpuNormals;
	delete[] cpuTangents;
	delete[] cpuBitangents;
	delete[] cpuTexCoords;
	delete[] cpuVertexIndices;
	delete[] vertices;
	delete[] triangles;
}

void release()
{
	cudaFree(gpuVertexIndices);
	cudaFree(gpuNormals);
	cudaFree(gpuTexCoords);
	cudaFree(gpuTriWoops);
	cudaFree(gpuTriIndices);
	cudaFree(gpuHDREnv);
	cudaFree(gpuNodes);
	cudaFree(gpuCamera);
	cudaFree(gpuAccumBuffer);
	cudaFree(gpuFrameBuffer);

	delete cpuCamera;
	delete interactiveCamera;
	delete gpuBVH;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	resetBuffer = true;
	switch (key)
	{
	case GLFW_KEY_ESCAPE:
		glfwSetWindowShouldClose(window, 0);
		break;
	case GLFW_KEY_SPACE:
		initCamera();
		break;
	case GLFW_KEY_A:
		if (action == GLFW_PRESS) interactiveCamera->slideDir.left = true;
		else if (action == GLFW_RELEASE) interactiveCamera->slideDir.left = false;
		break;
	case GLFW_KEY_D:
		if (action == GLFW_PRESS) interactiveCamera->slideDir.right = true;
		else if (action == GLFW_RELEASE) interactiveCamera->slideDir.right = false;
		break;
	case GLFW_KEY_W:
		if (action == GLFW_PRESS) interactiveCamera->slideDir.front = true;
		else if (action == GLFW_RELEASE) interactiveCamera->slideDir.front = false;
		break;
	case GLFW_KEY_S:
		if (action == GLFW_PRESS) interactiveCamera->slideDir.back = true;
		else if (action == GLFW_RELEASE) interactiveCamera->slideDir.back = false;
		break;
	case GLFW_KEY_G:
		interactiveCamera->changeAperDiam(0.1f);
		break;
	case GLFW_KEY_H:
		interactiveCamera->changeAperDiam(-0.1f);
		break;
	case GLFW_KEY_T:
		interactiveCamera->changeFocalDist(1.0f);
		break;
	case GLFW_KEY_Y:
		interactiveCamera->changeFocalDist(-1.0f);
		break;
	case GLFW_KEY_LEFT_SHIFT:
		if (action == GLFW_PRESS) interactiveCamera->setQuick(true);
		else if (action == GLFW_RELEASE) interactiveCamera->setQuick(false);
		break;
	default:
		resetBuffer = false;
		break;
	}
}

void mouseCursorCallback(GLFWwindow* window, double x, double y)
{
	if (lastX == 0 && lastY == 0)
	{
		lastX = x;
		lastY = y;
		return;
	}

	int deltaX = x - lastX;
	int deltaY = y - lastY;
	if ((deltaX != 0 || deltaY != 0) && mouseLeftButtonPressed)
	{
		interactiveCamera->changeYaw(deltaX * 0.015f);
		interactiveCamera->changePitch(-deltaY * 0.015f);
	}

	lastX = x;
	lastY = y;
	resetBuffer = mouseLeftButtonPressed;
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	mouseLeftButtonPressed = button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS;
}

void render()
{
	if (resetBuffer)
	{
		cudaMemset(gpuAccumBuffer, 1, WIDTH * HEIGHT * sizeof(Vec3f));
		frameNum = 0;
		resetBuffer = false;
	}
	frameNum++;

	interactiveCamera->extractCamera(cpuCamera);
	cudaMemcpy(gpuCamera, cpuCamera, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();

	cudaGLMapBufferObject((void **)&gpuFrameBuffer, vbo);
	glClear(GL_COLOR_BUFFER_BIT);

	cudaRender(gpuFrameBuffer, gpuAccumBuffer, gpuNodes, gpuCamera, frameNum, wangHash(frameNum));

	cudaDeviceSynchronize();

	cudaGLUnmapBufferObject(vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WIDTH * HEIGHT);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
}

int main()
{
	// init glfw
	GLFWwindow *window;
	if (!glfwInit())
	{
		printf("initialize glfw failed!\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(WIDTH, HEIGHT, "Cuda Path Tracer", nullptr, nullptr);
	if (!window)
	{
		printf("initialize window failed!\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	// init glew
	if (glewInit() != GLEW_OK)
	{
		printf("initialize glew failed!\n");
		return -1;
	}

	// define viewport
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	// init opengl
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, WIDTH, 0.0, HEIGHT);

	initVBO();

	// init camera
	initCamera();

	// init cuda
	initImage("media/textures/brick.jpg", cpuDiffuseImage, gpuDiffuseImage);
	initImage("media/textures/brick_normal.jpg", cpuNormalImage, gpuNormalImage);
	initHDR();
	initBVH();
	initCuda();
	initTextures(gpuVertexIndices, gpuNormals, gpuTangents, gpuBitangents, gpuTexCoords, trianglesNum, verticesNum,
		gpuTriWoops, gpuTriIndices, gpuHDREnv, triIndicesSize,
		gpuDiffuseImage, gpuNormalImage);

	// set callbacks
	glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mouseCursorCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		interactiveCamera->move();
		render();
		glfwSwapBuffers(window);
	}

	glfwTerminate();

	release();

	return 0;
}