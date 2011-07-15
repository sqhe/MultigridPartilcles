#include "StdAfx.h"
#include "mainDraw.h"

uint numParticles = 1<<16;

bool keyDown[256];

bool displaySliders = false;
int ox, oy;
int buttonState = -1;
int mode = 0;

ParticleSystem *particle_obj;
float zoom=0;
float slowdown=2.0f;	
float	xspeed=0;	
float	yspeed=0;	

GLuint point_sprite_texture;
GLuint sky_floor_texture;

Vector3 cameraPos(0,-5,-10);
Vector3 cameraRot(0,0,0);
Vector3 cameraPosLag(cameraPos);
Vector3 cameraRotLag(cameraRot);

const float walkSpeed=0.05;
const float rotateSpeed=0.002;
const float translateSpeed=0.0002;
const float inertia=0.1f;

float modelView[16];

float vv=10;
float s=20;

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
	if (!LoadGLTextures())
	{
		return;
	}
	

	particle_obj=new ParticleSystem(numParticles, false, false);
}

void cleanpaint()
{
	if (particle_obj)
	{
		delete particle_obj;

		particle_obj=NULL;
	}
}

void ixform(Vector3 &v, Vector3 &r, float *m)
{
	r.setX(v.x()*m[0] + v.y()*m[1] + v.z()*m[2]);
	r.setY(v.x()*m[4] + v.y()*m[5] + v.z()*m[6]);
	r.setZ(v.x()*m[8] + v.y()*m[9] + v.z()*m[10]);
}

void preDisplay()
{
	// move camera in view direction
	/*
	0   4   8   12  x
	1   5   9   13  y
	2   6   10  14  z
	*/
	
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

void display()
{
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.0);
	glDepthFunc(GL_LEQUAL);

	// draw floor

	glBindTexture(GL_TEXTURE_2D,sky_floor_texture);
	glEnable(GL_TEXTURE_2D);

	glColor3f(0,1,0);
	glNormal3f(0.0, 1.0, 0.0);

	glBegin(GL_QUADS);
	glTexCoord2f(0.f, 0.f);glVertex3f(-vv*s, -0, vv*s);                                                      
	glTexCoord2f(1.f, 0.f);glVertex3f(vv*s, -0, vv*s);                                                         
	glTexCoord2f(1.f, 1.f);glVertex3f(vv*s, -0, -vv*s);                                                          
	glTexCoord2f(0.f, 1.f);glVertex3f(-vv*s, -0, -vv*s);     
	glEnd();

	glDisable(GL_TEXTURE_2D);

	glColor3f(1,0,1);
	glBegin(GL_QUADS);
	glVertex3f(-10,-10,-10);
	glVertex3f(10,-10,-10);
	glVertex3f(10,10,-10);
	glVertex3f(-10,10,-10);

	glPushMatrix();

	glTranslatef(0,0,-1);

	glColor3f(0,1,1);

	/* Back side */
	glVertex3f(-1, -1, -1-1);
	glVertex3f(-1, 1, -1-1);
	glVertex3f(1, 1, -1-1);
	glVertex3f(1, -1, -1-1);

	/* Front side */
	glVertex3f(-1, -1, 1-1);
	glVertex3f(1, -1, 1-1);
	glVertex3f(1, 1, 1-1);
	glVertex3f(-1, 1, 1-1);

	/* Top side */
	glVertex3f(-1, 1, -1-1);
	glVertex3f(-1, 1, 1-1);
	glVertex3f(1, 1, 1-1);
	glVertex3f(1, 1, -1-1);

	/* Bottom side */
	glVertex3f(-1, -1, -1-1);
	glVertex3f(1, -1, -1-1);
	glVertex3f(1, -1, 1-1);
	glVertex3f(-1, -1, 1-1);

	/* Left side */
	glVertex3f(-1, -1, -1-1);
	glVertex3f(-1, -1, 1-1);
	glVertex3f(-1, 1, 1-1);
	glVertex3f(-1, 1, -1-1);

	/* Right side */
	glVertex3f(1, -1, -1-1);
	glVertex3f(1, 1, -1-1);
	glVertex3f(1, 1, 1-1);
	glVertex3f(1, -1, 1-1);

	glEnd();

	glPopMatrix();


	glBegin(GL_LINES);	
	// x axis
	glColor3f ( 1.0f, 0.0f, 0.0f);
	glVertex3f( 0.0f, 0.0f, 0.0f);
	glVertex3f( 1.0f, 0.0f, 0.0f);

	// y axis
	glColor3f ( 0.0f, 1.0f, 0.0f);
	glVertex3f( 0.0f, 0.0f, 0.0f);
	glVertex3f( 0.0f, 1.0f, 0.0f);

	// z axis
	glColor3f ( 0.0f, 0.0f, 1.0f);
	glVertex3f( 0.0f, 0.0f, 0.0f);
	glVertex3f( 0.0f, 0.0f, 1.0f);
	glEnd();

	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glBegin(GL_TRIANGLES);	

	// x axis arrow
	glColor3f ( 1.0f, 0.0f,  0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    0.0f,    0.04f);
	glVertex3f(0.9f,    0.028f,  0.028f);
	glVertex3f(0.9f,    0.0f,    0.04f);
	glVertex3f(0.9f,    0.028f,  0.028f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    0.028f,  0.028f);
	glVertex3f(0.9f,    0.04f,   0.0f);
	glVertex3f(0.9f,    0.028f,  0.028f);
	glVertex3f(0.9f,    0.04f,   0.0f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    0.04f,   0.0f);
	glVertex3f(0.9f,    0.028f,  -0.028f);
	glVertex3f(0.9f,    0.04f,   0.0f);
	glVertex3f(0.9f,    0.028f,  -0.028f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    0.028f,  -0.028f);
	glVertex3f(0.9f,    0.0f,    -0.04f);
	glVertex3f(0.9f,    0.028f,  -0.028f);
	glVertex3f(0.9f,    0.0f,    -0.04f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    0.0f,    -0.04f);
	glVertex3f(0.9f,    -0.028f, -0.028f);
	glVertex3f(0.9f,    0.0f,    -0.04f);
	glVertex3f(0.9f,    -0.028f, -0.028f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    -0.028f, -0.028f);
	glVertex3f(0.9f,    -0.04f,  0.0f);
	glVertex3f(0.9f,    -0.028f, -0.028f);
	glVertex3f(0.9f,    -0.04f,  0.0f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    -0.04f,  0.0f);
	glVertex3f(0.9f,    -0.028f, 0.028f);
	glVertex3f(0.9f,    -0.04f,  0.0f);
	glVertex3f(0.9f,    -0.028f, 0.028f);
	glVertex3f(0.9f,    0.0f,    0.0f);
	glVertex3f(1.0f,    0.0f,    0.0f);
	glVertex3f(0.9f,    -0.028f, 0.028f);
	glVertex3f(0.9f,    0.0f,    0.04f);
	glVertex3f(0.9f,    -0.028f, 0.028f);
	glVertex3f(0.9f,    0.0f,    0.04f);
	glVertex3f(0.9f,    0.0f,    0.0f);

	// y axis arrow
	glColor3f ( 0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(0.0f,    0.9f,    0.04f);
	glVertex3f(0.028f,  0.9f,    0.028f);
	glVertex3f(0.0f,    0.9f,    0.04f);
	glVertex3f(0.028f,  0.9f,    0.028f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(0.028f,  0.9f,    0.028f);
	glVertex3f(0.04f,   0.9f,    0.0f);
	glVertex3f(0.028f,  0.9f,    0.028f);
	glVertex3f(0.04f,   0.9f,    0.0f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(0.04f,   0.9f,    0.0f);
	glVertex3f(0.028f,  0.9f,    -0.028f);
	glVertex3f(0.04f,   0.9f,    0.0f);
	glVertex3f(0.028f,  0.9f,    -0.028f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(0.028f,  0.9f,    -0.028f);
	glVertex3f(0.0f,    0.9f,    -0.04f);
	glVertex3f(0.028f,  0.9f,    -0.028f);
	glVertex3f(0.0f,    0.9f,    -0.04f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(0.0f,    0.9f,    -0.04f);
	glVertex3f(-0.028f, 0.9f,    -0.028f);
	glVertex3f(0.0f,    0.9f,    -0.04f);
	glVertex3f(-0.028f, 0.9f,    -0.028f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(-0.028f, 0.9f,    -0.028f);
	glVertex3f(-0.04f,  0.9f,    0.0f);
	glVertex3f(-0.028f, 0.9f,    -0.028f);
	glVertex3f(-0.04f,  0.9f,    0.0f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(-0.04f,  0.9f,    0.0f);
	glVertex3f(-0.028f, 0.9f,    0.028f);
	glVertex3f(-0.04f,  0.9f,    0.0f);
	glVertex3f(-0.028f, 0.9f,    0.028f);
	glVertex3f(0.0f,    0.9f,    0.0f);
	glVertex3f(0.0f,    1.0f,    0.0f);
	glVertex3f(-0.028f, 0.9f,    0.028f);
	glVertex3f(0.0f,    0.9f,    0.04f);
	glVertex3f(-0.028f, 0.9f,    0.028f);
	glVertex3f(0.0f,    0.9f,    0.04f);
	glVertex3f(0.0f,    0.9f,    0.0f);

	// z axis arrow
	glColor3f ( 0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(0.0f,    0.04f,   0.9f);
	glVertex3f(0.028f,  0.028f,  0.9f);
	glVertex3f(0.0f,    0.04f,   0.9f);
	glVertex3f(0.028f,  0.028f,  0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(0.028f,  0.028f,  0.9f);
	glVertex3f(0.04f,   0.0f,    0.9f);
	glVertex3f(0.028f,  0.028f,  0.9f);
	glVertex3f(0.04f,   0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(0.04f,   0.0f,    0.9f);
	glVertex3f(0.028f,  -0.028f, 0.9f);
	glVertex3f(0.04f,   0.0f,    0.9f);
	glVertex3f(0.028f,  -0.028f, 0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(0.028f,  -0.028f, 0.9f);
	glVertex3f(0.0f,    -0.04f,  0.9f);
	glVertex3f(0.028f,  -0.028f, 0.9f);
	glVertex3f(0.0f,    -0.04f,  0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(0.0f,    -0.04f,  0.9f);
	glVertex3f(-0.028f, -0.028f, 0.9f);
	glVertex3f(0.0f,    -0.04f,  0.9f);
	glVertex3f(-0.028f, -0.028f, 0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(-0.028f, -0.028f, 0.9f);
	glVertex3f(-0.04f,  0.0f,    0.9f);
	glVertex3f(-0.028f, -0.028f, 0.9f);
	glVertex3f(-0.04f,  0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(-0.04f,  0.0f,    0.9f);
	glVertex3f(-0.028f, 0.028f,  0.9f);
	glVertex3f(-0.04f,  0.0f,    0.9f);
	glVertex3f(-0.028f, 0.028f,  0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);
	glVertex3f(0.0f,    0.0f,    1.0f);
	glVertex3f(-0.028f, 0.028f,  0.9f);
	glVertex3f(0.0f,    0.04f,   0.9f);
	glVertex3f(-0.028f, 0.028f,  0.9f);
	glVertex3f(0.0f,    0.04f,   0.9f);
	glVertex3f(0.0f,    0.0f,    0.9f);

	glEnd();



	//glBlendFunc(GL_SRC_ALPHA,GL_ONE/*GL_ONE_MINUS_SRC_ALPHA*/);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);

	glEnable(GL_BLEND);

	glAlphaFunc(GL_GREATER,0.1);


	glEnable(GL_ALPHA_TEST);



	glEnable(GL_POINT_SPRITE);
	glBindTexture(GL_TEXTURE_2D,point_sprite_texture);
	glTexEnvi(GL_POINT_SPRITE,GL_COORD_REPLACE,GL_TRUE);
	glEnable(GL_TEXTURE_2D);

	glEnable(GL_CULL_FACE);

	glPointSize(10.0);
/*
	for (int i=0;i<MAX_PARTICLES;i++)
	{
		if (particle_obj->particle[i].active)
		{
			float x=particle_obj->particle[i].x;
			float y=particle_obj->particle[i].y;
			float z=particle_obj->particle[i].z+zoom;

			glColor4f(particle_obj->particle[i].r,particle_obj->particle[i].g,particle_obj->particle[i].b,0.2*particle_obj->particle[i].life);

			//glPointSize(4.0);
			glBegin(GL_POINTS);
			glVertex3f(x,y,z);
			glEnd();

			particle_obj->particle[i].x+=1*particle_obj->particle[i].xi/(slowdown*1000);
			particle_obj->particle[i].y+=1*particle_obj->particle[i].yi/(slowdown*1000);
			particle_obj->particle[i].z+=particle_obj->particle[i].zi/(slowdown*1000);

			particle_obj->particle[i].life-=particle_obj->particle[i].fade;

			if (particle_obj->particle[i].life<0)
			{
				particle_obj->particle[i].life=1;
				particle_obj->particle[i].fade=float(rand()%100)/1000+0.003;

				particle_obj->particle[i].x=0;
				particle_obj->particle[i].y=0;
				particle_obj->particle[i].z=0;

				particle_obj->particle[i].xi=xspeed+float((rand()%60)-32);
				particle_obj->particle[i].yi=yspeed+float((rand()%60)-30);
				particle_obj->particle[i].zi=float((rand()%60)-30);

				particle_obj->particle[i].r=colors[1][0];
				particle_obj->particle[i].g=colors[1][1];
				particle_obj->particle[i].b=colors[1][2];
			}

		}
	}
*/
	glDisable(GL_TEXTURE_2D);
	glDisable(GL_POINT_SPRITE);
	glDisable(GL_BLEND);

	glFinish();
}
void reshape(int w,int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90.0, (float) w / (float) h, 0.01, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glViewport(0, 0, w, h);
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
		Vector3 v=Vector3(dx*translateSpeed,dy*translateSpeed,0);

		Vector3 r=Vector3(0,0,0);

		ixform(v,r,modelView);

		cameraPos+=r;
	}

	if (buttonState==GLUT_MIDDLE_BUTTON)
	{
		Vector3 v=Vector3(0,0,dy*translateSpeed);

		Vector3 r=Vector3(0,0,0);

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













