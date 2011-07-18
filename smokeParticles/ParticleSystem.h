#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "cudpp/cudpp.h"
#include "GpuArray.h"
#include "nvMath.h"
#include "cudpp/cudpp.h"

using namespace nv;

class ParticleSystem
{
public:
	ParticleSystem(uint numParticles, vec3f diskPosc, float radius, vec3f diskDirc, float vel_originalc,bool bUseVBO = true, bool bUseGL = true);
	~ParticleSystem();

	enum ParticleConfig
	{
		CONFIG_RANDOM,
		CONFIG_GRID,
		CONFIG_DISK,
		_NUM_CONFIGS
	};

	void step(float deltaTime);
	void depthSort();
	void reset(ParticleConfig config);

	uint getNumParticles(){ return m_numParticles; }

	uint getPosBuffer(){ return m_pos.getVbo(); }
	uint getVelBuffer(){ return m_vel.getVbo(); }
	uint getColorBuffer() { return 0; }
	uint getSortedIndexBuffer() { return m_indices.getVbo(); }
	uint *getSortedIndices();

	float getParticleRadius() { return m_particleRadius; }

	SimParams &getParams() { return m_params; }

	void setSorting(bool x) { m_doDepthSort = x; }
	void setModelView(float *m);
	void setSortVector(float3 v) { m_sortVector = v; }

	void addSphere(uint &index, vec3f pos, vec3f vel, int r, float spacing, float jitter, float lifetime);
	void discEmitter(uint &index, vec3f pos, vec3f vel, vec3f vx, vec3f vy, float r, int n, float lifetime, float lifetimeVariance);
	void sphereEmitter(uint &index, vec3f pos, vec3f vel, vec3f spread, float r, int n, float lifetime, float lifetimeVariance);

	void dumpParticles(uint start, uint count);
	void dumpBin(float4 **posData, float4 **velData);

	void setDisk(vec3f pos,float radius,vec3f dir);

protected:// method
	ParticleSystem() {}

	void _initialize(int numParticles, bool bUseGL=true);
	void _free();

	void initGrid(vec3f start, uint3 size,vec3f spacing, float jitter, vec3f vel, uint numParticles, float lifetime=100.0f);
	void initDiskRandom(vec3f diskPosc,vec3f diskDirc,float diskRadiusc, float vel_originalc,float lifetime=100.0f);
	void initCubeRandom(vec3f origin, vec3f size, vec3f vel, float lifetime=100.0f);

protected://data
	bool m_bInitialized;
	bool m_bUseVBO;
	uint m_numParticles;

	float m_particleRadius;

	GpuArray<float4> m_pos;
	GpuArray<float4> m_vel;

	SimParams m_params;

	float4x4 m_modelView;
	float3 m_sortVector;
	bool m_doDepthSort;

	CUDPPHandle m_sortHandle;
	GpuArray<float> m_sortKeys;
	GpuArray<uint> m_indices;

	uint m_timer;
	float m_time;

	vec3f diskPos;
	vec3f diskDir;

	float diskRadius;
	float vel_original;
};
#endif