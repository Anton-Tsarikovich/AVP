#include <iostream>
#include "TurnColorImage.cuh"
int main(int argc, char* argv[]) {
	TurnColorImage *im = new TurnColorImage();
	im->CPUReverse();
	im->CUDAReverse();
	delete im;
	return 0;
}