#include "rng.h"
#include <iostream>

void testCUDA(cudaError_t error, const char *file, int line)  {

	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value of the nv
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void mergeSmall_k(int* A, int* B, int* M, int lenA, int lenB){  
    int i = threadIdx.x;
    // extern __shared__ int** KP;
    int Kx, Ky;
    int Px, Py;
    int Qx, Qy;
    if (i<(lenA+lenB)) {
        if (i>lenA) {
            Kx = i-lenA;
            Ky = lenA;
            Px = lenA;
            Py = 1-lenA;
        } else {
            Kx = 0;
            Ky = i;
            Px = i;
            Py = 0;
        }
        while (true) {
            int offset = std::abs(Ky-Py)/2;
            Qx = Kx + offset;
            Qy = Ky - offset;
            if (Qy>=0 & Qx<=lenB & (Qx==lenB | Qy==0 | A[Qy]>B[Qx-1])) {
                if (Qx==lenB | Qy==0 | A[Qy-1]<=B[Qx]) {
                    if (Qy<lenA & (Qx==lenB | A[Qy]<=B[Qx])) {
                        M[i] = A[Qy];
                    } else {
                        M[i] = B[Qx];
                    }
                    break;
                } else {
                    Kx = Qx + 1;
                    Ky = Qy - 1;
                }
            } else {
                Px = Qx - 1;
                Py = Qy + 1;
            }
        }
    }
}

int main() {
    int LA = 281;
    int LB = 294;
    // int A[LA];
    // int B[LB];
    int *A,*B;
    A = (int*)malloc(LA*sizeof(int));
    B = (int*)malloc(LB*sizeof(int));
    
    int L = LA+LB;
    A[0] = 1;
    B[0] = 2;
    for (int i = 1; i<LA ; i++) {
        A[i] = A[i-1] + 5;
    }
    for (int i = 1; i<LB ; i++) {
        B[i] = B[i-1] + 3;
    }
    int M[L];
    int* MGPU;
    int* AGPU;
    int* BGPU;
    // for (int i=0; i<LA; ++i) {
    //     std::cout << A[i] << std::endl;
    // }
    testCUDA(cudaMalloc(&MGPU, L*sizeof(int)));
    testCUDA(cudaMalloc(&AGPU, LA*sizeof(int)));
    testCUDA(cudaMalloc(&BGPU, LB*sizeof(int)));

    testCUDA(cudaMemcpy(AGPU,A,LA*sizeof(int),cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(BGPU,B,LB*sizeof(int),cudaMemcpyHostToDevice));

    mergeSmall_k<<<1,1024>>>(AGPU,BGPU,MGPU,LA,LB);
    testCUDA(cudaMemcpy(M, MGPU, L*sizeof(int), cudaMemcpyDeviceToHost));
    for (int i=0; i<L; ++i) {
        std::cout << M[i] << std::endl;
    }
    testCUDA(cudaFree(MGPU));
    testCUDA(cudaFree(AGPU));
    testCUDA(cudaFree(BGPU));

}