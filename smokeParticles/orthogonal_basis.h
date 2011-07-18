#ifndef _ORTHOGONAL_BASIS_
#define _ORTHOGONAL_BASIS_

typedef unsigned int uint;

#include <GL/glew.h>
#include "nvMath.h"


using namespace nv;

vec3f tcross(vec3f a,vec3f b);

void find_orth_basic(vec3f Dir,float * &bas);

#endif