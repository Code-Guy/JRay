#ifndef CAMERA_H
#define CAMERA_H

#include "Util.h"

struct Camera
{
	Vec2f res;
	Vec3f pos;
	Vec3f view;
	Vec3f up;
	Vec2f fov;
	float aperRad;
	float focalDist;
};

class InteractiveCamera
{
public:
	struct SlideDir
	{
		bool front = false;
		bool back = false;
		bool left = false;
		bool right = false;
	};

	InteractiveCamera();
	virtual ~InteractiveCamera();

	void setRes(float x, float y);
	void setFovx(float fovx);
	void setQuick(bool flag);

	void changeYaw(float val);
	void changePitch(float val);
	void changeFocalDist(float val);
	void changeAperDiam(float val);

	void move();

	void extractCamera(Camera* camera);

	Vec2f res;
	Vec2f fov;

	SlideDir slideDir;

private:
	Vec3f pos;
	Vec3f view;

	float yaw;
	float pitch;

	float aperRad;
	float focalDist;

	bool quick;
};

#endif // CAMERA_H