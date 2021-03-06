#pragma once

#include "MainFrm.h"

#include <GL/glaux.h>

AUX_RGBImageRec *LoadBMP(char *Filename);
int LoadGLTextures();
void initpaint();
void cleanpaint();
void ixform(vec3f &v, vec3f &r, float *m);
void preDisplay();
void display();
void renderscene();
void reshape(int w,int h);
void motion(int x, int y);
void key(unsigned char key, int /*x*/, int /*y*/);
void keyUp(unsigned char key, int /*x*/, int /*y*/);
void mouse(int button, int state, int x, int y);
void drawbackground();
void drawboundbox();
void renderparticles();
