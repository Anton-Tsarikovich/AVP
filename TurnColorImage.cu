#include "TurnColorImage.cuh"


template < typename T > 
__device__ void swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


__global__ void ReverceGPU(UINT * deviceImage, const int Height, const int Stride, int HeightOnTwo) {

	int i = blockIdx.x * 8 + threadIdx.x;
	int j = blockIdx.y * 32 + threadIdx.y;

	if (i >= Stride || j >= HeightOnTwo) {
		return;
	}
	swap(deviceImage[(Height - j) * Stride - Stride + i], deviceImage[j * Stride  + i]);


}

TurnColorImage::TurnColorImage() {
	GdiplusStartup(&GDIPlusToken, &GDIInput, NULL);
	bitMapCPU = new Bitmap(L"inputImage.jpg");
	bitMapGPU = new Bitmap(L"inputImage.jpg");
	bitmapDataCPU = new BitmapData;
	bitmapDataGPU = new BitmapData;
	bitMapCPU->LockBits(NULL, ImageLockMode::ImageLockModeWrite, PixelFormat24bppRGB, bitmapDataCPU);
	bitMapGPU->LockBits(NULL, ImageLockMode::ImageLockModeWrite, PixelFormat24bppRGB, bitmapDataGPU);
	pixCPU = (UINT*)bitmapDataCPU->Scan0;
	pixGPU = (UINT*)bitmapDataGPU->Scan0;
	cudaMalloc((void**)&deviceImage, bitmapDataGPU->Height * bitmapDataGPU->Stride / 4 * sizeof(UINT));
	cudaMemcpy(deviceImage, pixGPU, bitmapDataGPU->Height * bitmapDataGPU->Stride / 4 * sizeof(UINT), cudaMemcpyHostToDevice);
}
TurnColorImage::~TurnColorImage(){
	delete bitmapDataCPU;
	delete bitmapDataGPU;
	delete bitMapCPU;
	delete bitMapGPU;
	GdiplusShutdown(GDIPlusToken);
}

void TurnColorImage::check() {
	bool compareFlag = false;
	for (auto i = 0; i < bitmapDataGPU->Height; i++) {
		for (auto j = 0; j < bitmapDataGPU->Stride / 4; j++) {
			if (pixCPU[i * bitmapDataGPU->Stride / 4 + j] != pixGPU[i * bitmapDataGPU->Stride / 4 + j]) {
				compareFlag = true;
				std::cout << i << " " << j << std::endl;
			}
		}
	}
	std::cout << ((compareFlag) ? "Images are not equals" : "Images are equals") << std::endl;
}


void TurnColorImage::CPUReverse() {

	auto startTick = __rdtsc();
	if (bitmapDataCPU->Height % 2 == 0) {
		for (int i = bitmapDataCPU->Height / 2 - 1, j = bitmapDataCPU->Height / 2; i >= 0; i--, j++) {
			for (auto k = 0; k < bitmapDataCPU->Stride / 4; k++) {
				std::swap(pixCPU[i * bitmapDataCPU->Stride / 4 + k],
					pixCPU[j * bitmapDataCPU->Stride / 4 + k]);
			}
		}
	}
	else {
		for (int i = bitmapDataCPU->Height / 2 - 1, j = bitmapDataCPU->Height / 2 + 1; i >= 0; i--, j++) {
			for (auto k = 0; k < bitmapDataCPU->Stride / 4; k++) {
				std::swap(pixCPU[i * bitmapDataCPU->Stride / 4 + k],
					pixCPU[j * bitmapDataCPU->Stride / 4 + k]);
			}
		}
	}
	auto finishTick = __rdtsc();
	std::cout << "Run Time CPU = " << double((finishTick - startTick) / CLK_TCK / 1000) << std::endl;
	bitMapCPU->UnlockBits(bitmapDataCPU);
	CLSID pngClsid;
	GetEncoderClsid(L"image/jpeg", &pngClsid);
	bitMapCPU->Save(L"CPU_out.jpg", &pngClsid, NULL);


}

void TurnColorImage::CUDAReverse() {
	cudaEvent_t start, stop;
	float timer = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);


	dim3 threads(8, 32);
	dim3 blocks(ceil((bitmapDataGPU->Stride + threads.x  + 1) / 8), ceil((bitmapDataGPU->Height  + threads.x * 2 + 1) / 64));

	cudaEventSynchronize(start);

	ReverceGPU <<< blocks, threads >>>  (deviceImage, bitmapDataGPU->Height, bitmapDataGPU->Stride / 4, bitmapDataGPU->Height / 2);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cout << cudaGetErrorString(error) << std::endl;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);


	cudaMemcpy(pixGPU, deviceImage, bitmapDataGPU->Height * bitmapDataGPU->Stride / 4 * sizeof(UINT), cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&timer, start, stop);
	std::cout << "Run Time GPU = " << timer << std::endl;
	bitMapGPU->UnlockBits(bitmapDataGPU);
	CLSID pngClsid;
	GetEncoderClsid(L"image/jpeg", &pngClsid);
	bitMapGPU->Save(L"GPU_out.jpg", &pngClsid, NULL);
	check();
}

int TurnColorImage::GetEncoderClsid(const WCHAR* format, CLSID* pClsid)
{
	UINT  num = 0;          // number of image encoders
	UINT  size = 0;         // size of the image encoder array in bytes

	ImageCodecInfo* pImageCodecInfo = NULL;

	GetImageEncodersSize(&num, &size);
	if (size == 0)
		return -1;  // Failure

	pImageCodecInfo = (ImageCodecInfo*)(malloc(size));
	if (pImageCodecInfo == NULL)
		return -1;  // Failure

	GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			free(pImageCodecInfo);
			return j;  // Success
		}
	}

	free(pImageCodecInfo);
	return -1;  // Failure
}
