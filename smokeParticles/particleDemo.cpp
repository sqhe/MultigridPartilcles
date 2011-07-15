#include "stdafx.h"
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <math.h>

#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#if defined (_WIN32)
#include <GL/wglew.h>
#endif

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>

#include "ParticleSystem.h"
#include "Particlesystem.cuh"
#include "SmokeRenderer.h"
#include "paramgl.h"
#include "GLSLProgram.h"
#include "SmokeShaders.h"

//uint numParticles = 1<<16;
//
//ParticleSystem *psystem=0;
//SmokeRenderer *renderer=0;
//GLSLProgram *floorProg=0;
//
//int width=1280,height=1024;
//
//int ox,oy;
//int buttonState=0;
//bool keyDown[256];
//
//vec3f cameraPos(0, -1, -4);
//vec3f cameraRot(0, 0, 0);
//vec3f cameraPosLag(cameraPos);
//vec3f cameraRotLag(cameraRot);
//vec3f cursorPos(0, 1, 0);
//vec3f cursorPosLag(cursorPos);
//
//vec3f lightPos(5.0, 5.0, -5.0);
//
//const float inertia = 0.1f;
//const float translateSpeed = 0.002f;
//const float cursorSpeed = 0.01f;
//const float rotateSpeed = 0.2f;
//const float walkSpeed = 0.05f;
//
//enum { M_VIEW = 0, M_MOVE_CURSOR, M_MOVE_LIGHT };
//int mode = 0;
int displayMode = (int) SmokeRenderer::VOLUMETRIC;

