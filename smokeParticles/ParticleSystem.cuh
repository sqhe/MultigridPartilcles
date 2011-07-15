#include "particles_kernel.cuh"

extern "C"
{
void initCuda(bool bUseGL);
void setParameters(SimParams *hostParams);
void createNoiseTexture(int w,int h,int d);

void
integrateSystem(float4 *oldPos, float4 *newPos,
				float4 *oldVel, float4 *newVel,
				float deltaTime,
				int numParticles);
				
void
calcDepth(float4* pos,
		  float* keys,
		  uint* indices,
		  float3 sortVector,
		  int numParticles);
}