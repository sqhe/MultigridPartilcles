// ChildView.cpp : implementation of the CChildView class
//

#include "stdafx.h"
#include "Simplifier.h"
#include "ChildView.h"

#include "mainDraw.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

int winWidth = 1280, winHeight = 1024;
int timer_span=30,timer_span_bak=10;

// CChildView

CChildView::CChildView()
: m_pDC(NULL)
{
}

CChildView::~CChildView()
{
}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_CREATE()
	ON_WM_DESTROY()
	ON_WM_SIZE()
	ON_WM_ERASEBKGND()
	ON_WM_KEYDOWN()
	ON_WM_KEYUP()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_MBUTTONDOWN()
	ON_WM_MBUTTONUP()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_SETCURSOR()
	ON_WM_TIMER()
	ON_WM_MOUSEMOVE()

	ON_COMMAND(ID_M_VIEW, OnMoveView)
	ON_COMMAND(ID_M_SPHERE, OnMoveSphere)
	ON_COMMAND(ID_M_CHIMNEY, OnMoveChimney)
	ON_COMMAND(ID_DIS_BOUND, OnShowBound)

END_MESSAGE_MAP()



// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.style |= WS_CLIPSIBLINGS | WS_CLIPCHILDREN ;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
		::LoadCursor(NULL, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW+1), NULL);

	return TRUE;
}

void CChildView::OnMoveView()
{
	mode = M_VIEW;
}

void CChildView::OnMoveSphere()
{
	mode = M_MOVE_SPHERE;
}

void CChildView::OnMoveChimney()
{
	mode = M_MOVE_CHIMNEY;
}

void CChildView::OnShowBound()
{
	displayBound = !displayBound;
}

void CChildView::OnPaint() 
{
	CPaintDC dc(this); // device context for painting
	
	// TODO: Add your message handler code here

	preDisplay();

	DrawScene();

	SwapBuffers(wglGetCurrentDC());
	
	// Do not call CWnd::OnPaint() for painting messages
}


void CChildView::Init()
{
	PIXELFORMATDESCRIPTOR pfd;
	int n;
	HGLRC hrc;

	m_pDC=new CClientDC(this);

	ASSERT(m_pDC != NULL);

	if(!SetThePixelFormat())
		return;

	n=::GetPixelFormat(m_pDC->GetSafeHdc());

	::DescribePixelFormat(m_pDC->GetSafeHdc(), n,sizeof(pfd),&pfd);

	hrc=wglCreateContext(m_pDC->GetSafeHdc());
	wglMakeCurrent(m_pDC->GetSafeHdc(),hrc);

	//particle_obj=new Particlesysterm();
	SetTimer( TIMER_ANIMATE/*TIMER_TOOLBAR*/, timer_span/*TOOLBAR_RATE*/, NULL);
	initpaint();
}

void CChildView::clean()
{
	cleanpaint();
}

void CChildView::DrawAxes()
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	glScalef(1.4f, 1.4f, 1.4f);

	glLineWidth(4);

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


	// 	glPolygonMode(GL_FRONT,GL_LINE);
	// 	glPolygonMode(GL_BACK,GL_LINE);
	glEnd();
	glPopMatrix();
	//glEnable(GL_LIGHTING);
	glColor3f ( 1.0f, 1.0f, 0.0f);

	glLineWidth(1);
}

BOOL CChildView::SetThePixelFormat(void)
{
	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),	// size of this pfd
		1,                              // version number
		PFD_DRAW_TO_WINDOW |            // support window
		PFD_SUPPORT_OPENGL |            // support OpenGL
		PFD_DOUBLEBUFFER	|			// double buffered
		PFD_STEREO,               
		PFD_TYPE_RGBA,                  // RGBA type
		24,                             // 24-bit color depth
		0, 0, 0, 0, 0, 0,               // color bits ignored
		0,                              // no alpha buffer
		0,                              // shift bit ignored
		0,                              // no accumulation buffer
		0, 0, 0, 0,                     // accum bits ignored
		32,                             // 32-bit z-buffer	
		0,                              // no stencil buffer
		0,                              // no auxiliary buffer
		PFD_MAIN_PLANE,                 // main layer
		0,                              // reserved
		0, 0, 0                         // layer masks ignored
	};

	int pixelformat;
	if((pixelformat = ChoosePixelFormat(m_pDC->GetSafeHdc(), &pfd)) == 0)
	{
		MessageBox( "ChoosePixelFormat failed", "Error", MB_OK);
		return FALSE;
	}

	if(SetPixelFormat(m_pDC->GetSafeHdc(), pixelformat, &pfd) == FALSE)
	{
		MessageBox( "SetPixelFormat failed", "Error", MB_OK);
		return FALSE;
	}

	// pCDC, iPixelFormat, pfd are declared in above code

	pixelformat = GetPixelFormat (m_pDC->GetSafeHdc());

	DescribePixelFormat (m_pDC->GetSafeHdc(), pixelformat, sizeof
		(PIXELFORMATDESCRIPTOR), &pfd);

	return TRUE;
}

void CChildView::DrawScene(void)
{
	display();

	if (displayBound)
		DrawAxes();
}

int CChildView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	// TODO:  Add your specialized creation code here

	Init();

	return 0;
}

void CChildView::OnDestroy()
{
	CWnd::OnDestroy();

	// TODO: Add your message handler code here

	HGLRC hrc;
	hrc = ::wglGetCurrentContext();
	if (hrc)
		::wglMakeCurrent(NULL,NULL);
	if (m_pDC)
		delete m_pDC;

}

void CChildView::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	// TODO: Add your message handler code here
	winWidth=cx;
	winHeight=cy;
	Project();
}

BOOL CChildView::OnEraseBkgnd(CDC* pDC)
{
	// TODO: Add your message handler code here and/or call default

	return TRUE;
}

void CChildView::Project()
{
	RECT	rect;

	GetClientRect( &rect);
	GLsizei nWidth = rect.right;
	GLsizei nHeight = rect.bottom;
	reshape(nWidth, nHeight);
}

void CChildView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnKeyDown(nChar, nRepCnt, nFlags);
	
	m_bKeyDown=TRUE;
	char nChar2=nChar;
	if (!GetAsyncKeyState(VK_LSHIFT)&&nChar2<='Z'&&nChar2>='A')
	{
		nChar2+='a'-'A';
	}
	key(nChar2,0,0);
	Invalidate();
}

void CChildView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnKeyUp(nChar, nRepCnt, nFlags);

	m_bKeyDown=FALSE;
	char nChar2=nChar;
	if (!GetAsyncKeyState(VK_LSHIFT)&&nChar2<='Z'&&nChar2>='A')
	{
		nChar2+='a'-'A';
	}
	keyUp(nChar2,0,0);
	Invalidate();
}

void CChildView::OnLButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnLButtonDown(nFlags, point);

	m_bLButtonDown=TRUE;
	mouse(GLUT_LEFT_BUTTON, 0, point.x, point.y);	
	Invalidate();
}

void CChildView::OnLButtonUp(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnLButtonUp(nFlags, point);

	m_bLButtonDown=FALSE;
	mouse(GLUT_LEFT_BUTTON, 1, point.x, point.y);
	Invalidate();
}

void CChildView::OnMButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnMButtonDown(nFlags, point);

	mouse(GLUT_MIDDLE_BUTTON, 0, point.x, point.y);
	Invalidate();
}

void CChildView::OnMButtonUp(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnMButtonUp(nFlags, point);

	mouse(GLUT_MIDDLE_BUTTON, 1, point.x, point.y);
	Invalidate();
}

void CChildView::OnRButtonDown(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnRButtonDown(nFlags, point);

	mouse(GLUT_RIGHT_BUTTON, 0, point.x, point.y);
	Invalidate();
}

void CChildView::OnRButtonUp(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnRButtonUp(nFlags, point);

	mouse(GLUT_RIGHT_BUTTON, 1, point.x, point.y);
	Invalidate();
}

BOOL CChildView::OnSetCursor(CWnd* pWnd, UINT nHitTest, UINT message)
{
	// TODO: Add your message handler code here and/or call default

	//return CWnd::OnSetCursor(pWnd, nHitTest, message);

	HCURSOR hCursor;

	switch (m_nMouseAct)
	{
	case MOUSE_SPIN:
		hCursor = AfxGetApp()->LoadCursor( IDC_SPIN );
		SetCursor( hCursor );
		break;
	case MOUSE_ZOOM:
		hCursor = AfxGetApp()->LoadCursor( IDC_ZOOM );
		SetCursor( hCursor );
		break;
	case MOUSE_TRANSLATE:
		hCursor = AfxGetApp()->LoadCursor( IDC_TRANSLATE );
		SetCursor( hCursor );
		break;
	default:
		hCursor = AfxGetApp()->LoadCursor( IDC_ARROW );
		SetCursor( hCursor );
		break;
	}

	return CWnd::OnSetCursor(pWnd, nHitTest, message);

	return TRUE;
}

void CChildView::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: Add your message handler code here and/or call default

	CWnd::OnTimer(nIDEvent);

	Invalidate();
}

void CChildView::OnMouseMove(UINT nFlags, CPoint point)
{
	// TODO: Add your message handler code here and/or call default

	motion(point.x,point.y);

	CWnd::OnMouseMove(nFlags, point);

	Invalidate();
}
