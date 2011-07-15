#ifndef UCDAVIS_RENDER_BUFFER_H
#define UCDAVIS_RENDER_BUFFER_H

#include "framebufferObject.h"

class Renderbuffer
{
public:
	Renderbuffer();
	Renderbuffer(GLenum internalFormat, int width, int height);
	~Renderbuffer();

	void Bind();
	void Unbind();
	void Set(GLenum internalFormat, int width, int height);
	GLuint GetId() const;

	static GLint GetMaxSize();

private:
	GLuint m_bufId;
	static GLuint _CreateBufferId();
};

#endif