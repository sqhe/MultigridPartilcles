// Simplifier.h : main header file for the Simplifier application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CSimplifierApp:
// See Simplifier.cpp for the implementation of this class
//

class CSimplifierApp : public CWinApp
{
public:
	CSimplifierApp();


// Overrides
public:
	virtual BOOL InitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CSimplifierApp theApp;