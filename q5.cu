#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void MergeSmallBatch_k(float* Input, size_t sizeM, float* Output, int d)
{
	//Example：Block
	int i = threadIdx.x % d; // ith element in d
	int j = (threadIdx.x - i) / d; //jth AB in block
	int k = j + blockIdx.x * (blockDim.x / d); //kth AB in total

	const size_t sizeA = d / 2;
	const size_t sizeB = d / 2;

	Input = Input + k * d;

	float* A = Input;
	float* B = A + sizeA;

	int offset;
	int Kx, Ky;
	int Px, Py;
	int Qx, Qy;

	if (i > sizeA) {
		Kx = i - sizeA;
		Ky = sizeA;
		Px = sizeA;
		Py = i - sizeA;
	}
	else {
		Kx = 0;
		Ky = i;
		Px = i;
		Py = 0;
	}

	while (true) {
		int offset = std::abs(Ky - Py) / 2;
		Qx = Kx + offset;
		Qy = Ky - offset;
		if ((Qy >= 0) && (Qx <= sizeB) &&
			((Qy == sizeA) || (Qx == 0) || (A[Qy] > B[Qx - 1]))) {

			if ((Qx == sizeB) || (Qy == 0) || (A[Qy - 1] <= B[Qx]))
			{
				if ((Qy < sizeA) && ((Qx == sizeB) || (A[Qy] <= B[Qx])))
				{
					Output[i + k * d] = A[Qy];
				}
				else
				{
					Output[i + k * d] = B[Qx];
				}
				break;
			}
			else
			{
				Kx = Qx + 1; Ky = Qy - 1;
			}
		}
		else {
			Px = Qx - 1; Py = Qy + 1;
		}
	}
}

int main() {
	const int Nsamples = 256;
	cudaError_t cudaStatus;

	float* InputHost;
	float* OutputHost;
	float* InputCuda;
	float* OutputCuda;

	cudaStatus = cudaSetDevice(0);
	for (int d = 2; d <= 4096; d *= 2) {
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		InputHost = (float*)malloc(Nsamples * d * sizeof(float));
		OutputHost = (float*)malloc(Nsamples * d * sizeof(float));

		size_t sizeA = d / 2;
		size_t sizeB = d / 2;
		for (int i = 0; i < Nsamples * d; i++) {
			if (i % d == 0 || i % d == sizeA) {
				InputHost[i] = (rand() % 30) * 1.0;
			}
			else {
				InputHost[i] = (rand() % 30) * 1.0 + InputHost[i - 1];
			}
		}

		cudaEventRecord(start);
		cudaStatus = cudaMalloc((void**)&InputCuda, Nsamples * d * sizeof(float));
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "InputCuda内存分配失败");

		cudaStatus = cudaMalloc((void**)&OutputCuda, Nsamples * d * sizeof(float));
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "OutputCuda内存分配失败");


		cudaStatus = cudaMemcpy(InputCuda, InputHost, Nsamples * d * sizeof(float), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "从CPU复制数据失败!");
		}

		MergeSmallBatch_k << <256, 1024 >> > (InputCuda, Nsamples * d, OutputCuda, d);

		cudaStatus = cudaMemcpy(OutputHost, OutputCuda, Nsamples * d * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "从GPU复制数据失败!");
		}
		cudaEventRecord(end);
		cudaEventSynchronize(end);

		float t{ 0 };
		cudaEventElapsedTime(&t, start, end);

		cudaEventDestroy(start);
		cudaEventDestroy(end);

		//for (int i = 0; i < 2 * d; ++i) {
		//	std::cout << OutputHost[i] << std::endl;
		//}

		free(InputHost);
		free(OutputHost);
		if (InputCuda != NULL) { cudaFree(InputCuda); InputCuda = NULL; }
		if (OutputCuda != NULL) { cudaFree(OutputCuda); OutputCuda = NULL; }

		printf("For Length %d, Time has passed %f \n", d, t);
	}


	return 0;
}