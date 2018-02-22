/*  Aditya Subramanian
 *  Muralidaran
 *  adityasu
 */

#ifndef A3_HPP
#define A3_HPP
#include <cuda_runtime_api.h>
#include <math.h>

__global__ void compute_kernalFunction(float a, int a_idx, int n, float h, float *x1, float *y1);

void gaussian_kde(int n, float h, const std::vector<float>& x, std::vector<float>& y) {

	// Device Copies of x & y vector
	float *x1;
	float *y_out;

    // Allocate Memory in GPU
    cudaMalloc((void **)&x1, n*sizeof(float));
    cudaMalloc((void **)&y_out, n*sizeof(float));

    cudaMemcpy(x1, &x[0], n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_out, &y[0], n*sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i<n; i++) {
    	compute_kernalFunction <<<(n+255)/256, 256, 256*sizeof(float)>>> (x[i], i, n, h, x1, y_out);
    }

	cudaThreadSynchronize();
	cudaMemcpy(&y[0], y_out, n*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(x1);
	cudaFree(y_out);

} // gaussian_kde

// Function to run on GPGPU 
__global__ void compute_kernalFunction(float a, int a_idx, int n, float h, float *x1, float *y1) {
	extern __shared__  float blocksum[];

	int tid = threadIdx.x;
	int x_idx = blockIdx.x * blockDim.x + tid;

	// Kernal function calculation
	float x_p = (a - x1[x_idx]) / h;
	float k_val = (1/pow(2*3.14,0.5)) * exp(-(pow(x_p,2))/2);
	blocksum[tid] = k_val;

	__syncthreads();

	// Summation of kernal_function(a)
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) blocksum[tid] += blocksum[tid + s];
		__syncthreads();
	}

	// Writing to output of density_function(a)
	if(tid == 0) {
		y1[a_idx] += (blocksum[0] * (1/(n*h)));

	}

	
}
#endif // A3_HPP
