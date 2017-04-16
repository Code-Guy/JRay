#include "camera.h"

const float SlideSpeed = 1.0f;

InteractiveCamera::InteractiveCamera()
{
	pos = Vec3f(-50, 80, 200);
	yaw = 345.0f;
	pitch = 19.0f;
	//aperRad = 1.6f;
	aperRad = 0.0f;
	focalDist = 150.0f;

	res = Vec2f(1280, 720);
	fov = Vec2f(45, 45);

	quick = false;
}

InteractiveCamera::~InteractiveCamera()
{

}

void InteractiveCamera::setRes(float x, float y)
{
	res = Vec2f(x, y);
	setFovx(fov.x);
}

void InteractiveCamera::setFovx(float fovx)
{
	fov.x = fovx;
	fov.y = degree(atan(tan(radian(fovx) / 2.0f) * (res.y / res.x)) * 2.0f);
}

void InteractiveCamera::setQuick(bool flag)
{
	quick = flag;
}

void InteractiveCamera::changeYaw(float val)
{
	yaw += val;
	yaw = mod(yaw, 360.0f);
	//printf("pos: %.4f %.4f %.4f\n", pos.x, pos.y, pos.z);
	//printf("yaw: %.4f\n", yaw);
}

void InteractiveCamera::changePitch(float val)
{
	pitch += val;
	pitch = clampf(pitch, -89.0f, 89.0f);
	//printf("pos: %.4f %.4f %.4f\n", pos.x, pos.y, pos.z);
	//printf("pitch: %.4f\n", pitch);
}

void InteractiveCamera::changeFocalDist(float val)
{
	focalDist += val;
	//printf("focalDist: %.4f\n", focalDist);
}

void InteractiveCamera::changeAperDiam(float val)
{
	aperRad += (aperRad + 0.01f) * val;
}

void InteractiveCamera::move()
{
	Vec3f right = cross(view, Vec3f(0.0f, 1.0f, 0.0f)); 
	right.normalize();

	float speed = quick ? SlideSpeed * 3.0f : SlideSpeed;
	if (slideDir.front)
	{
		pos += view * speed;
	}
	if (slideDir.back)
	{
		pos -= view * speed;
	}
	if (slideDir.left)
	{
		pos -= right * speed;
	}
	if (slideDir.right)
	{
		pos += right * speed;
	}
}

void InteractiveCamera::extractCamera(Camera* camera)
{
	view.x = sin(radian(yaw)) * cos(radian(pitch));
	view.y = sin(radian(pitch));
	view.z = cos(radian(yaw)) * cos(radian(pitch));
	view *= -1.0f;

	camera->pos = pos;
	camera->view = view;
	camera->up = Vec3f(0, 1, 0);
	camera->res = Vec2f(res.x, res.y);
	camera->fov = Vec2f(fov.x, fov.y);
	camera->aperRad = aperRad;
	camera->focalDist = focalDist;
}