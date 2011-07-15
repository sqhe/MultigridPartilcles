#ifndef PARTICLES_KERNEL_H
#define PARTICLES_KERNEL_H

#ifdef USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#include "vector_types.h"
typedef unsigned int uint;

struct SimParams {
	float3 gravity;
	float globalDamping;
	float noiseFreq;
	float noiseAmp;
	float3 cursorPos;
	
	float time;
	float3 noiseSpeed;
};

struct float4x4 {
	float m[16];
};

#endif