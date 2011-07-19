#include "StdAfx.h"
#include "mainDraw.h"

#include "GLSLProgram.h"
#include "SmokeShaders.h"
#include "SmokeRenderer.h"

#define M_PI		3.141592653589793238462643383280

uint numParticles = 1<<12;

float vel_orignal = 0.1;

bool keyDown[256];

float boundBox[6];
int gridResolution[3];
float dir_size[3];

bool displaySliders = false;

int ox, oy;
int buttonState = -1;

int mode = 0;
bool displayBound = true;

ParticleSystem *particle_obj = 0;
SmokeRenderer *renderer = 0;
GLSLProgram  *backgroundProg = 0;

float   zoom=0;
float   slowdown=2.0f;	
float	xspeed=0;	
float	yspeed=0;	

GLuint point_sprite_texture;
GLuint sky_floor_texture;

vec3f cameraPos(0,-5,-10);
vec3f cameraRot(0,0,0);
vec3f cameraPosLag(cameraPos);
vec3f cameraRotLag(cameraRot);

vec3f chimneyPos(0,0,0);
vec3f chimneyDir(0,1,0);
vec3f obspherePos(0,6,0);

GLfloat obsphereRadius=2;
GLfloat chimneyupRadius=1;
GLfloat chimneydownRadius=1;
GLfloat chimneyHeight=2;

vec3f diskPos;

float spspeed = 0.2;
float chspeed = 0.2;

GLUquadric *chimneyObj;
GLUquadric *obsphereObj;

const float walkSpeed=0.05;
const float rotateSpeed=0.002;
const float translateSpeed=0.0002;
const float inertia=0.1f;

float modelView[16];

float vv=10;
float s=20;

float timestep = 0.5f;
float spriteSize = 0.05f;
float alpha = 0.1;
float shadowAlpha = 0.02;
bool displayLightBuffer = false;
float blurRadius = 2.0;
int numSlices = 64;
int numDisplayedSlices = numSlices;
bool sort = true;

vec3f lightPos(5.0, 5.0, -5.0);
vec3f lightColor(1.0, 1.0, 0.8);
vec3f colorAttenuation(0.5, 0.75, 1.0);

AUX_RGBImageRec *LoadBMP(char *Filename)
{
	FILE *File=NULL;

	if (!Filename)
	{
		return NULL;
	}

	File=fopen(Filename,"r");

	if (File)
	{
		fclose(File);

		return auxDIBImageLoad(Filename);
	}

	return NULL;
}

int LoadGLTextures()
{
	int Status=FALSE;

	AUX_RGBImageRec *TextureImage[1];

	memset(TextureImage,0,sizeof(void *)*1);

	if (TextureImage[0]=LoadBMP("Data/Particle.bmp"))
	{
		Status=TRUE;

		glGenTextures(1,&point_sprite_texture);

		glBindTexture(GL_TEXTURE_2D,point_sprite_texture);
		glTexImage2D(GL_TEXTURE_2D,0,3,TextureImage[0]->sizeX,TextureImage[0]->sizeY,0,GL_RGB,GL_UNSIGNED_BYTE,TextureImage[0]->data);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	}

	if (TextureImage[0]=LoadBMP("Data/skyfloortile.bmp"))
	{
		Status=TRUE;

		glGenTextures(1,&sky_floor_texture);

		glBindTexture(GL_TEXTURE_2D,sky_floor_texture);
		glTexImage2D(GL_TEXTURE_2D,0,3,TextureImage[0]->sizeX,TextureImage[0]->sizeY,0,GL_RGB,GL_UNSIGNED_BYTE,TextureImage[0]->data);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	}

	if (TextureImage[0])
	{
		if (TextureImage[0]->data)
		{
			free(TextureImage[0]->data);
		}
		
		free(TextureImage[0]);
	}

	return Status;
}

void initpaint()
{
	glewInit();

	if (!LoadGLTextures())
	{
		return;
	}

	diskPos = chimneyPos + chimneyHeight*chimneyDir;
	
	particle_obj=new ParticleSystem(numParticles,diskPos,chimneyupRadius,chimneyDir,vel_orignal, true, true);
	particle_obj->reset(ParticleSystem::CONFIG_DISK);

	renderer = new SmokeRenderer(numParticles);
	//renderer->setLightTarget(vec3f(0,1,0));

	chimneyObj = gluNewQuadric();
	obsphereObj = gluNewQuadric();

	backgroundProg = new GLSLProgram(floorVS,floorPS);

	boundBox[0] = -5 ;
	boundBox[1] = 5 ;
	boundBox[2] = 0 ;
	boundBox[3] = 10 ;
	boundBox[4] = -5 ;
	boundBox[5] = 5 ;

	gridResolution[0] = 32 ;
	gridResolution[1] = 32 ;
	gridResolution[2] = 32 ;

	for (int i=0;i<3;i++)
		dir_size[i]=(boundBox[2*i+1]-boundBox[2*i])/gridResolution[i];
}

void cleanpaint()
{
	if (particle_obj)
	{
		delete particle_obj;

		particle_obj=NULL;
	}

	gluDeleteQuadric(chimneyObj);
	gluDeleteQuadric(obsphereObj);

	if (backgroundProg)
		delete backgroundProg;

	if(renderer)
	{
		delete renderer;
		renderer = NULL;
	}
}

void ixform(vec3f &v, vec3f &r, float *m)
{
	r.x = v.x*m[0] + v.y*m[1] + v.z*m[2];
	r.y = v.x*m[4] + v.y*m[5] + v.z*m[6];
	r.z = v.x*m[8] + v.y*m[9] + v.z*m[10];
}

void preDisplay()
{
	// move camera in view direction
	/*
	0   4   8   12  x
	1   5   9   13  y
	2   6   10  14  z
	*/

	switch(mode)
	{
	case M_VIEW:
		{
			if (keyDown['w']) {
				cameraPos[0] += modelView[2] * walkSpeed;
				cameraPos[1] += modelView[6] * walkSpeed;
				cameraPos[2] += modelView[10] * walkSpeed;
			}
			if (keyDown['s']) {
				cameraPos[0] -= modelView[2] * walkSpeed;
				cameraPos[1] -= modelView[6] * walkSpeed;
				cameraPos[2] -= modelView[10] * walkSpeed;
			}
			if (keyDown['a']) {
				cameraPos[0] += modelView[0] * walkSpeed;
				cameraPos[1] += modelView[4] * walkSpeed;
				cameraPos[2] += modelView[8] * walkSpeed;
			}
			if (keyDown['d']) {
				cameraPos[0] -= modelView[0] * walkSpeed;
				cameraPos[1] -= modelView[4] * walkSpeed;
				cameraPos[2] -= modelView[8] * walkSpeed;
			}
			if (keyDown['e']) {
				cameraPos[0] += modelView[1] * walkSpeed;
				cameraPos[1] += modelView[5] * walkSpeed;
				cameraPos[2] += modelView[9] * walkSpeed;
			}
			if (keyDown['q']) {
				cameraPos[0] -= modelView[1] * walkSpeed;
				cameraPos[1] -= modelView[5] * walkSpeed;
				cameraPos[2] -= modelView[9] * walkSpeed;
			}
		}
		break;

	case M_MOVE_SPHERE:
		{
			if (keyDown['w']) {
				obspherePos[0] -= modelView[2] * spspeed;
				obspherePos[1] -= modelView[6] * spspeed;
				obspherePos[2] -= modelView[10] * spspeed;
			}
			if (keyDown['s']) {
				obspherePos[0] += modelView[2] * spspeed;
				obspherePos[1] += modelView[6] * spspeed;
				obspherePos[2] += modelView[10] * spspeed;
			}
			if (keyDown['a']) {
				obspherePos[0] -= modelView[0] * spspeed;
				obspherePos[1] -= modelView[4] * spspeed;
				obspherePos[2] -= modelView[8] * spspeed;
			}
			if (keyDown['d']) {
				obspherePos[0] += modelView[0] * spspeed;
				obspherePos[1] += modelView[4] * spspeed;
				obspherePos[2] += modelView[8] * spspeed;
			}
			if (keyDown['e']) {
				obspherePos[0] -= modelView[1] * spspeed;
				obspherePos[1] -= modelView[5] * spspeed;
				obspherePos[2] -= modelView[9] * spspeed;
			}
			if (keyDown['q']) {
				obspherePos[0] += modelView[1] * spspeed;
				obspherePos[1] += modelView[5] * spspeed;
				obspherePos[2] += modelView[9] * spspeed;
			}
		}
		break;

	case M_MOVE_CHIMNEY:
		{
			if (keyDown['w']) {
				chimneyPos[0] -= modelView[2] * chspeed;
				chimneyPos[1] -= modelView[6] * chspeed;
				chimneyPos[2] -= modelView[10] * chspeed;
			}
			if (keyDown['s']) {
				chimneyPos[0] += modelView[2] * chspeed;
				chimneyPos[1] += modelView[6] * chspeed;
				chimneyPos[2] += modelView[10] * chspeed;
			}
			if (keyDown['a']) {
				chimneyPos[0] -= modelView[0] * chspeed;
				chimneyPos[1] -= modelView[4] * chspeed;
				chimneyPos[2] -= modelView[8] * chspeed;
			}
			if (keyDown['d']) {
				chimneyPos[0] += modelView[0] * chspeed;
				chimneyPos[1] += modelView[4] * chspeed;
				chimneyPos[2] += modelView[8] * chspeed;
			}
			if (keyDown['e']) {
				chimneyPos[0] -= modelView[1] * chspeed;
				chimneyPos[1] -= modelView[5] * chspeed;
				chimneyPos[2] -= modelView[9] * chspeed;
			}
			if (keyDown['q']) {
				chimneyPos[0] += modelView[1] * chspeed;
				chimneyPos[1] += modelView[5] * chspeed;
				chimneyPos[2] += modelView[9] * chspeed;
			}

			diskPos = chimneyPos + chimneyHeight*chimneyDir;
			particle_obj->setDisk(diskPos,chimneyupRadius,chimneyDir);
			particle_obj->reset(ParticleSystem::CONFIG_DISK);
		}
		break;
	default:
		{
			if (keyDown['w']) {
				cameraPos[0] += modelView[2] * walkSpeed;
				cameraPos[1] += modelView[6] * walkSpeed;
				cameraPos[2] += modelView[10] * walkSpeed;
			}
			if (keyDown['s']) {
				cameraPos[0] -= modelView[2] * walkSpeed;
				cameraPos[1] -= modelView[6] * walkSpeed;
				cameraPos[2] -= modelView[10] * walkSpeed;
			}
			if (keyDown['a']) {
				cameraPos[0] += modelView[0] * walkSpeed;
				cameraPos[1] += modelView[4] * walkSpeed;
				cameraPos[2] += modelView[8] * walkSpeed;
			}
			if (keyDown['d']) {
				cameraPos[0] -= modelView[0] * walkSpeed;
				cameraPos[1] -= modelView[4] * walkSpeed;
				cameraPos[2] -= modelView[8] * walkSpeed;
			}
			if (keyDown['e']) {
				cameraPos[0] += modelView[1] * walkSpeed;
				cameraPos[1] += modelView[5] * walkSpeed;
				cameraPos[2] += modelView[9] * walkSpeed;
			}
			if (keyDown['q']) {
				cameraPos[0] -= modelView[1] * walkSpeed;
				cameraPos[1] -= modelView[5] * walkSpeed;
				cameraPos[2] -= modelView[9] * walkSpeed;
			}
		}
		break;
	}

	cameraPosLag+=(cameraPos-cameraPosLag)*inertia;
	cameraRotLag+=(cameraRot-cameraRotLag)*inertia;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glRotatef(cameraRotLag[0],1,0,0);
	glRotatef(cameraRotLag[1],0,1,0);

	glTranslatef(cameraPosLag[0],cameraPosLag[1],cameraPosLag[2]);

	glGetFloatv(GL_MODELVIEW_MATRIX,modelView);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void drawbackground()
{
	// draw boundbox

	if (displayBound)
		drawboundbox();

	//vec4f tempss(1,1,1,1);
	// use shader
	backgroundProg->enable();
	backgroundProg->bindTexture("tex",sky_floor_texture,GL_TEXTURE_2D,0);
	backgroundProg->bindTexture("shadowTex",sky_floor_texture,GL_TEXTURE_2D,1);
	backgroundProg->setUniformfv("lightPosEye",vec4f(1,1,1,1),3);
	backgroundProg->setUniformfv("lightColor",lightColor,3);

	// draw floor

	//glColor3f(1.0, 1.0, 1.0);
	glColor3f(0.75,0.75,0.75);
	glNormal3f(0.0, 1.0, 0.0);
	glBegin(GL_QUADS);
	{
		float s = 100.f;
		float rep = 20.f;
		glTexCoord2f(0.f, 0.f); glVertex3f(-s, 0, -s);
		glTexCoord2f(rep, 0.f); glVertex3f(s, 0, -s);
		glTexCoord2f(rep, rep); glVertex3f(s, 0, s);
		glTexCoord2f(0.f, rep); glVertex3f(-s, 0, s);
	}
	glEnd();

	// draw obstacle sphere

	glPushMatrix();

	glTranslatef(obspherePos[0],obspherePos[1],obspherePos[2]);

	glColor3f(0.8,0.8,0.2);
	gluSphere(obsphereObj,obsphereRadius,50,50);

	glPopMatrix();

	// draw chimney

	glPushMatrix();

	glTranslatef(chimneyPos[0],chimneyPos[1],chimneyPos[2]);
	glRotatef(-90,1,0,0);

	glColor3f(0.3,0.3,0.3);
	gluCylinder(chimneyObj,chimneydownRadius,chimneyupRadius,chimneyHeight,50,50);

	glPopMatrix();

	backgroundProg->disable();
}

void drawboundbox()
{
	glColor3f(0,0,1);

	glLineWidth(4);

	glBegin(GL_LINES);

	glVertex3f(boundBox[1],boundBox[2],boundBox[5]);
	glVertex3f(boundBox[1],boundBox[2],boundBox[4]);

	glVertex3f(boundBox[1],boundBox[2],boundBox[4]);
	glVertex3f(boundBox[0],boundBox[2],boundBox[4]);

	glVertex3f(boundBox[0],boundBox[2],boundBox[4]);
	glVertex3f(boundBox[0],boundBox[2],boundBox[5]);

	glVertex3f(boundBox[0],boundBox[2],boundBox[5]);
	glVertex3f(boundBox[1],boundBox[2],boundBox[5]);

	glVertex3f(boundBox[1],boundBox[3],boundBox[5]);
	glVertex3f(boundBox[1],boundBox[3],boundBox[4]);

	glVertex3f(boundBox[1],boundBox[3],boundBox[4]);
	glVertex3f(boundBox[0],boundBox[3],boundBox[4]);

	glVertex3f(boundBox[0],boundBox[3],boundBox[4]);
	glVertex3f(boundBox[0],boundBox[3],boundBox[5]);

	glVertex3f(boundBox[0],boundBox[3],boundBox[5]);
	glVertex3f(boundBox[1],boundBox[3],boundBox[5]);

	glVertex3f(boundBox[1],boundBox[2],boundBox[5]);
	glVertex3f(boundBox[1],boundBox[3],boundBox[5]);

	glVertex3f(boundBox[1],boundBox[2],boundBox[4]);
	glVertex3f(boundBox[1],boundBox[3],boundBox[4]);

	glVertex3f(boundBox[0],boundBox[2],boundBox[4]);
	glVertex3f(boundBox[0],boundBox[3],boundBox[4]);

	glVertex3f(boundBox[0],boundBox[2],boundBox[5]);
	glVertex3f(boundBox[0],boundBox[3],boundBox[5]);

	glEnd();

	glLineWidth(1);
}

void renderparticles()
{
	//particle_obj->step(timestep);

	renderer->calcVectors();
	vec3f sortVector = renderer->getSortVector();

	particle_obj->setSortVector(make_float3(sortVector.x,sortVector.y,sortVector.z));
	particle_obj->setModelView(modelView);
	particle_obj->setSorting(sort);
	particle_obj->depthSort();

	renderer->beginSceneRender(SmokeRenderer::LIGHT_BUFFER);
	renderscene();
	renderer->endSceneRender(SmokeRenderer::LIGHT_BUFFER);

	renderer->beginSceneRender(SmokeRenderer::SCENE_BUFFER);
	renderscene();
	renderer->endSceneRender(SmokeRenderer::SCENE_BUFFER);

	renderer->setPositionBuffer(particle_obj->getPosBuffer());
	renderer->setVelocityBuffer(particle_obj->getVelBuffer());
	renderer->setIndexBuffer(particle_obj->getSortedIndexBuffer());

	renderer->setNumParticles(particle_obj->getNumParticles());
	renderer->setParticleRadius(spriteSize);
	renderer->setDisplayLightBuffer(displayLightBuffer);
	renderer->setAlpha(alpha);
	renderer->setShadowAlpha(shadowAlpha);
	renderer->setLightPosition(lightPos);
	renderer->setColorAttenuation(colorAttenuation);
	renderer->setLightColor(lightColor);
	renderer->setNumSlices(numSlices);
	renderer->setNumDisplayedSlices(numDisplayedSlices);
	renderer->setBlurRadius(blurRadius);

	renderer->render();

	particle_obj->step(timestep);
}

void renderscene()
{
	glClearColor(0,0,0,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	drawbackground();
}

void display()
{
	renderscene();

	renderparticles();
}
void reshape(int w,int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90.0, (float) w / (float) h, 0.01, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0, 0, w, h);

	renderer->setFOV(90.0);
	renderer->setWindowSize(w,h);
}

void motion(int x, int y)
{
	float dx,dy;

	dx=x-ox;
	dy=y-oy;

	if (buttonState==GLUT_LEFT_BUTTON)
	{
		cameraRot[0]+=dy*rotateSpeed;
		cameraRot[1]+=dx*rotateSpeed;
	}

	if (buttonState==GLUT_RIGHT_BUTTON)
	{
		vec3f v=vec3f(dx*translateSpeed,dy*translateSpeed,0);

		vec3f r=vec3f(0,0,0);

		ixform(v,r,modelView);

		cameraPos+=r;
	}

	if (buttonState==GLUT_MIDDLE_BUTTON)
	{
		vec3f v=vec3f(0,0,dy*translateSpeed);

		vec3f r=vec3f(0,0,0);

		ixform(v,r,modelView);

		cameraPos+=r;
	}
}

void key(unsigned char key, int /*x*/, int /*y*/)
{
	keyDown[key] = true;
}

void keyUp(unsigned char key, int /*x*/, int /*y*/)
{
	keyDown[key] = false;
}

void mouse(int button, int state, int x, int y)
{
	if (state==GLUT_DOWN)
	{
		buttonState=button;
	}
	else if (state==GLUT_UP)
	{
		buttonState=-1;
	}

	ox=x;
	oy=y;
}













