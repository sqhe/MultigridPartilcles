#include "stdafx.h"
#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <GL/glew.h>

#if defined(_APPLE_) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <cutil_inline.h>

#include "ParticleSystem.h"
#include "ParticleSystem.cuh"
#include "particles_kernel.cuh"

#include "orthogonal_basis.h"


//********************************************************
#include <fstream>
//********************************************************

#ifndef CUDART_PI_F
#define CUDART_PI_F			3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numParticles,vec3f diskPosc, float radius, vec3f diskDirc, float vel_originalc, bool bUseVBO /* = true */, bool bUseGL /* = true */) :
	m_bInitialized(false),
	m_bUseVBO(bUseVBO),
	m_numParticles(numParticles),
	m_particleRadius(0.1),
	m_doDepthSort(false),
	m_timer(0),
	m_time(0.0f)
{
	m_params.gravity = make_float3(0.0f, 0.0f, 0.0f);
	m_params.globalDamping = 1.0f;
	m_params.noiseSpeed = make_float3(0.0f, 0.0f, 0.0f);

	_initialize(numParticles, bUseGL);

	diskPos = diskPosc;
	diskDir = diskDirc;
	diskRadius = radius;
	vel_original = vel_originalc;
}

ParticleSystem::~ParticleSystem()
{
	_free();
	m_numParticles=0;
}

void
ParticleSystem::setDisk(vec3f pos,float radius,vec3f dir)
{
	diskPos = pos;
	diskRadius = radius;
	diskDir = dir;
}

void
ParticleSystem::_initialize(int numParticles, bool bUseGL)
{
	assert(!m_bInitialized);

	initCuda(bUseGL);
	createNoiseTexture(64,64,64);

	m_numParticles=numParticles;

	m_pos.alloc(m_numParticles,m_bUseVBO,true);
	m_vel.alloc(m_numParticles,m_bUseVBO,true);


	CUDPPConfiguration sortConfig;
	sortConfig.algorithm=CUDPP_SORT_RADIX;
	sortConfig.datatype=CUDPP_FLOAT;
	sortConfig.op=CUDPP_ADD;
	sortConfig.options=CUDPP_OPTION_KEY_VALUE_PAIRS;
	cudppPlan(&m_sortHandle,sortConfig,m_numParticles,1,0);

	m_sortKeys.alloc(m_numParticles);
	m_indices.alloc(m_numParticles,m_bUseVBO,false,true);

	cutilCheckError(cutCreateTimer(&m_timer));
	setParameters(&m_params);

	m_bInitialized=true;
}

void
ParticleSystem::_free()
{
	assert(m_bInitialized);

	cudppDestroyPlan(m_sortHandle);
}

void
ParticleSystem::step(float deltaTime)
{
	assert(m_bInitialized);

	m_params.time=m_time;
	setParameters(&m_params);

	m_pos.map();
	m_vel.map();

	integrateSystem(m_pos.getDevicePtr(),m_pos.getDeviceWritePtr(),
					m_vel.getDevicePtr(),m_vel.getDeviceWritePtr(),
					deltaTime,
					m_numParticles);

	m_pos.unmap();
	m_vel.unmap();

	m_pos.swap();
	m_vel.swap();

	m_time += deltaTime;
}

void
ParticleSystem::depthSort()
{
	if (!m_doDepthSort)
	{
		return;
	}

	m_pos.map();
	m_indices.map();

	calcDepth(m_pos.getDevicePtr(),m_sortKeys.getDevicePtr(),m_indices.getDevicePtr(),m_sortVector,m_numParticles);

	cudppSort(m_sortHandle,m_sortKeys.getDevicePtr(),m_indices.getDevicePtr(),32,m_numParticles);

	m_pos.unmap();
	m_indices.unmap();
}

uint *
ParticleSystem::getSortedIndices()
{
	// copy sorted indices back to CPU
	m_indices.copy(GpuArray<uint>::DEVICE_TO_HOST);
	return m_indices.getHostPtr();
}

inline float frand()
{
	return rand()/(float)RAND_MAX;
}

inline float sfrand()
{
	return frand()*2.0f-1.0f;
}

inline vec3f svrand()
{
	return vec3f(sfrand(),sfrand(),sfrand());
}

inline vec2f randCircle()
{
	vec2f r;
	do 
	{
		r=vec2f(sfrand(),sfrand());
	} while (length(r)>1.0f);
	return r;
}

inline vec3f randSphere()
{
	vec3f r;
	do {
		r = svrand();
	} while(length(r) > 1.0f);
	return r;
}

void
ParticleSystem::initGrid(vec3f start, uint3 size,vec3f spacing, float jitter, vec3f vel, uint numParticles, float lifetime/* =100.0f */)
{
	srand(1973);

	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	for (uint z=0;z<size.z;z++)
	{
		for (uint y=0;y<size.y;y++)
		{
			for (uint x=0;x<size.x;x++)
			{
				uint i=(z*size.y*size.x) + (y*size.x) + x;

				if (i<numParticles)
				{
					vec3f pos= start + spacing*vec3f(x,y,z)+svrand()*jitter;

					posPtr[i] = make_float4(pos.x,pos.y,pos.z,0.0f);
					velPtr[i] = make_float4(vel.x,vel.y,vel.z,lifetime);
				}
			}
		}
	}
}

void
ParticleSystem::initDiskRandom(vec3f diskPosc, vec3f diskDirc,float diskRadiusc, float vel_originalc,float lifetime/*=100.0f*/)
{
	//diskPos = diskPosc;
	//diskDir = diskDirc;
	//diskRadius = diskRadiusc;
	setDisk(diskPosc,diskRadiusc,diskDirc);
	vel_original = vel_originalc;

	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	float *bas;

	bas =new float[6];

	find_orth_basic(diskDir,bas);

	vec3f pdir0(bas[0],bas[1],bas[2]);
	vec3f pdir1(bas[2],bas[4],bas[5]);

	for (uint i=0;i<m_numParticles;i++)
	{
		//float b0,b1;
		//b0 = sfrand(); b1 = sfrand();
		//if(b0*b0+b1*b1 > 1)
		//{
		//	b0*=0.707;
		//	b1*=0.707;
		//}
		//vec3f pos = diskPos + diskRadius*b0*pdir0 + diskRadius*b1*pdir1;

		float r,theta;
		r = frand(); theta = 3.141592653589793238462643383280 * sfrand();
		vec3f pos = diskPos + r*diskRadius*(pdir0*sin(theta) + pdir1*cos(theta));
		
		vec3f vel = vel_original*diskDir;
		posPtr[i] = make_float4(pos.x,pos.y,pos.z,0.0f);
		velPtr[i] = make_float4(vel.x,vel.y,vel.z,lifetime);
	}

	//*****************************************************************************************
	std::ofstream ofile("pos.txt");

	ofile<<"init"<<std::endl;
	for (uint i=0;i<m_numParticles;i++)
	{
		ofile<<i<<": "<<posPtr[i].x<<" "<<posPtr[i].y<<" "<<posPtr[i].z<<" "<<posPtr[i].w<<"  ,        "<<velPtr[i].x<<" "<<velPtr[i].y<<" "<<velPtr[i].z<<" "<<velPtr[i].w<<" "<<std::endl;
	}

	ofile<<std::endl;

	ofile.close();
	//*****************************************************************************************

}

void
ParticleSystem::initCubeRandom(vec3f origin, vec3f size, vec3f vel, float lifetime/* =100.0f */)
{
	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	for(uint i=0; i < m_numParticles; i++) 
	{
		vec3f pos = origin + svrand()*size;
		posPtr[i] = make_float4(pos.x, pos.y, pos.z, 0.0f);
		velPtr[i] = make_float4(vel.x, vel.y, vel.z, lifetime);
	}
}

void
ParticleSystem::addSphere(uint &index, vec3f pos, vec3f vel, int r, float spacing, float jitter, float lifetime)
{
	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	uint start = index;
	uint count = 0; 
	for(int z=-r; z<=r; z++) {
		for(int y=-r; y<=r; y++) {
			for(int x=-r; x<=r; x++) {
				vec3f delta = vec3f(x, y, z)*spacing;
				float dist = length(delta);
				if ((dist <= spacing*r) && (index < m_numParticles)) {
					vec3f p = pos + delta + svrand()*jitter;

					posPtr[index] = make_float4(pos.x, pos.y, pos.z, 0.0f);
					velPtr[index] = make_float4(vel.x, vel.y, vel.z, lifetime);

					index++;
					count++;
				}
			}
		}
	}

	m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
	m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void
ParticleSystem::reset(ParticleConfig config)
{
	switch(config)
	{
	default:
	case CONFIG_DISK:
		initDiskRandom(diskPos,diskDir,diskRadius,vel_original,100.0);
		break;

	case CONFIG_RANDOM:
		initCubeRandom(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 1.0, 1.0), vec3f(0.0f), 100.0);
		break;

	case CONFIG_GRID:
		{
			float jitter = m_particleRadius*0.01f;
			uint s = (int) ceilf(powf((float) m_numParticles, 1.0f / 3.0f));
			uint gridSize[3];
			gridSize[0] = gridSize[1] = gridSize[2] = s;
			initGrid(vec3f(-1.0, 0.0, -1.0), make_uint3(s, s, s), vec3f(m_particleRadius*2.0f), jitter, vec3f(0.0), m_numParticles, 100.0);
		}
		break;
	}

	m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE);
	m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE);
}

void
ParticleSystem::discEmitter(uint &index, vec3f pos, vec3f vel, vec3f vx, vec3f vy, float r, int n, float lifetime, float lifetimeVariance)
{
	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	uint start = index; 
	uint count = 0;
	for(int i=0; i<n; i++) {
		vec2f delta = randCircle() * r;
		if (index < m_numParticles) {
			vec3f p = pos + delta.x*vx + delta.y*vy;
			float lt = lifetime + frand()*lifetimeVariance;

			posPtr[index] = make_float4(p.x, p.y, p.z, 0.0f);
			velPtr[index] = make_float4(vel.x, vel.y, vel.z, lt);

			index++;
			count++;
		}
	}

	m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
	m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void
ParticleSystem::sphereEmitter(uint &index, vec3f pos, vec3f vel, vec3f spread, float r, int n, float lifetime, float lifetimeVariance)
{
	float4 *posPtr = m_pos.getHostPtr();
	float4 *velPtr = m_vel.getHostPtr();

	uint start = index; 
	uint count = 0;
	for(int i=0; i<n; i++) {
		vec3f x = randSphere();
		//float dist = length(x);
		if (index < m_numParticles) {

			vec3f p = pos + x*r;
			float age = 0.0;

			float lt = lifetime + frand()*lifetimeVariance;

			vec3f dir = randSphere();
			dir.y = fabs(dir.y);
			vec3f v = vel + dir*spread;

			posPtr[index] = make_float4(p.x, p.y, p.z, age);
			velPtr[index] = make_float4(v.x, v.y, v.z, lt);

			index++;
			count++;
		}
	}

	m_pos.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
	m_vel.copy(GpuArray<float4>::HOST_TO_DEVICE, start, count);
}

void
ParticleSystem::setModelView(float *m)
{
	for(int i=0; i<16; i++) {
		m_modelView.m[i] = m[i];
	}
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
	m_pos.copy(GpuArray<float4>::DEVICE_TO_HOST);
	float4 *pos = m_pos.getHostPtr();

	m_vel.copy(GpuArray<float4>::DEVICE_TO_HOST);
	float4 *vel = m_vel.getHostPtr();

	for(uint i=start; i<start+count; i++) {
		printf("%d: ", i);
		printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", pos[i].x, pos[i].y, pos[i].z, pos[i].w);
		printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", vel[i].x, vel[i].y, vel[i].z, vel[i].w);
	}
}

void
ParticleSystem::dumpBin( float4 **posData,
						float4 **velData)
{
	m_pos.copy(GpuArray<float4>::DEVICE_TO_HOST);
	*posData = m_pos.getHostPtr();

	m_vel.copy(GpuArray<float4>::DEVICE_TO_HOST);
	*velData = m_vel.getHostPtr();
}
