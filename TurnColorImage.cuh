#include <iostream>
#include <Windows.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <gdiplus.h>	
#pragma comment(lib, "Gdiplus.lib")

using namespace Gdiplus;

class TurnColorImage {
public:
	TurnColorImage();
	~TurnColorImage();
	void CPUReverse();
	void CUDAReverse();

private:
	GdiplusStartupInput GDIInput;
	DWORD_PTR GDIPlusToken;
	Bitmap *bitMapCPU;
	Bitmap *bitMapGPU;
	BitmapData *bitmapDataCPU;
	BitmapData *bitmapDataGPU;
	UINT *pixCPU;
	UINT *pixGPU;
	UINT *deviceImage;


	int GetEncoderClsid(const WCHAR* format, CLSID* pClsid);
	void check();


};