#include <cuda.h>
#include <stdio.h>


__global__ void GetWeightKernel(float *input, int input_len, float *addr,
		float *exclusive_weight, int num_of_exclusive_weight,
		int *page_table_addr, int page_size, int num_of_weight_page, int start, int end)
{
	int idx, page_num, page, offset;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < input_len;
			i += blockDim.x * gridDim.x) {
		idx = start+i;
		page_num = idx / page_size;
		page = page_table_addr[page_num];
		offset = idx % page_size;

		if (page < num_of_weight_page) {
			input[i] = addr[page*page_size + offset];
		} else {
			input[i] = exclusive_weight[(page-num_of_weight_page)*page_size + offset];
		}
	}
}

extern "C" {
float *LoadExclusiveWeight(float *exclusive_weight_page, int num_of_exclusive_weight)
{
	float *exclusive_weight;
	cudaMalloc(&exclusive_weight, sizeof(float)*num_of_exclusive_weight);
	cudaMemcpy(exclusive_weight, exclusive_weight_page,
		sizeof(float)*num_of_exclusive_weight,
		cudaMemcpyHostToDevice);

	return exclusive_weight;
}

void FreeExclusiveWeight(float *exclusive_weight)
{
	cudaFree(exclusive_weight);
}

void GetWeightKernelLauncher(float *input, int input_len,
		float *exclusive_weight, int num_of_exclusive_weight,
		float* addr,
		int* page_table_addr, int page_size, int num_of_weight_page, int start, int end)
{
	GetWeightKernel<<<32, 256>>>(input, input_len,
			addr, exclusive_weight, num_of_exclusive_weight,
			page_table_addr, page_size, num_of_weight_page, start, end);
	cudaDeviceSynchronize();
}
}
