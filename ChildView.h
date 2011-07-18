// ChildView.h : interface of the CChildView class
//


#pragma once

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/wglew.h>

#include "ParticleSystem.h"

#define	MOUSE_SPIN		1
#define	MOUSE_TRANSLATE	2
#define	MOUSE_ZOOM		3

#define	TIMER_ANIMATE	1
#define	TIMER_TOOLBAR	2
#define	TIMER_SIMULATION	3

#define	ANIMATE_RATE	50
#define	TOOLBAR_RATE	20
#define	ROTATE_RATE		4.0f
#define	ZOOM_RATE		0.05f

extern int mode;
extern bool displayBound;

enum M_MODE { M_VIEW = 0, M_MOVE_SPHERE, M_MOVE_CHIMNEY };

// CChildView window

class CChildView : public CWnd
{
// Construction
public:
	CChildView();

// Attributes
public:
	BOOL m_bKeyDown;
	BOOL m_bLButtonDown;
	int  m_nMouseAct;

// Operations
public:

// Overrides
	protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CChildView();

	// Generated message map functions
protected:
	afx_msg void OnPaint();
	DECLARE_MESSAGE_MAP()

	afx_msg void OnMoveView();
	afx_msg void OnMoveSphere();
	afx_msg void OnMoveChimney();
	afx_msg void OnShowBound();
public:
	CClientDC *m_pDC;
	void DrawAxes();
	void Init();
	void clean();
	BOOL SetThePixelFormat(void);
	void DrawScene(void);
	void Project();
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnDestroy();
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg BOOL OnEraseBkgnd(CDC* pDC);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags);
	afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnMButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnMButtonUp(UINT nFlags, CPoint point);
	afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
	afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
	afx_msg BOOL OnSetCursor(CWnd* pWnd, UINT nHitTest, UINT message);
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnMouseMove(UINT nFlags, CPoint point);
};

