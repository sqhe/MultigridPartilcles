#include "stdafx.h" 
#include "orthogonal_basis.h"

vec3f tcross(vec3f a,vec3f b)
{
	return vec3f(a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]);
}
void find_orth_basic(vec3f Dir,float * & bas)
{
	float len = dot(Dir,Dir);

	bas[0] = 0;
	bas[1] = 0;
	bas[2] = 0;
	bas[3] = 0;
	bas[4] = 0;
	bas[5] = 0;

	if (len <= 1e-8) 
		return ;

	len = sqrt(len);

	vec3f nor = Dir/len;

	vec3f x(1,0,0);
	vec3f y(0,1,0);
	vec3f z(0,0,1);

	vec3f temp;

	temp = tcross(Dir,x);

	len = dot(temp,temp);

	if (len < 1e-8)
	{
		bas[1] = 1;
		bas[5] = 1;

		return ;
	}

	float k;

	k=dot(Dir,x)/dot(Dir,Dir);

	vec3f b1;
	vec3f b2;

	b1=x-k*Dir;

	len = sqrt(dot(b1,b1));

	b1=b1/len;
	
	for (int i=0;i<3;i++)
		bas[i]=b1[i];

	float len2;

	//temp = tcross(Dir,y);

	len = dot(Dir,y);

	len2 = dot(b1,y);

	//len2 = dot(temp,temp);

	if ((len < 1e-8) && (len2 < 1e-8))
	{
		//b2 = y;
		bas[4] = 1;
		return;
	}

	float k0,k1;

	if (fabs(b1[0]*Dir[2]-b1[2]*Dir[0]) > 1e-6)
	{
		k0=nor[1];
		k1=b1[1];

		b2=y-k0*nor-k1*b1;

		len = sqrt(dot(b2,b2));

		b2=b2/len;

		for (int i=0;i<3;i++)
			bas[i+3]=b2[i];

		return ;
	}

	k0=nor[2];
	k1=b1[2];

	b2=z-k0*nor-k1*b1;

	len = sqrt(dot(b2,b2));

	b2=b2/len;

	for (int i=0;i<3;i++)
		bas[i+3]=b2[i];

	return;
}