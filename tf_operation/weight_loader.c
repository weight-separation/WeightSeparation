#include <stdio.h>


float *LoadExclusiveWeight(float *exclusive_weight_page, int num_of_exclusive_weight);
void FreeExclusiveWeight(float *exclusive_weight);
void GetWeightKernelLauncher(float *input, int input_len,
		float *exclusive_weight, int num_of_exclusive_weight,
		float* addr,
		int* page_table_addr, int page_size, int num_of_weight_page, int start, int end);

void get_weight(long long int *weight_address_list, int *weight_len_list,
		float *exclusive_weight_page, int num_of_exclusive_weight,
		int num_of_weight,
		long long int virtual_weight_address, long long int page_table_address,
		int page_size, int num_of_weight_page)
{
	int start = 0;
	int end = 0;

	float *exclusive_weight = LoadExclusiveWeight(exclusive_weight_page,
		num_of_exclusive_weight);

	for (int i = 0; i < num_of_weight; i++) {
		float *input = (float *)weight_address_list[i];
		int input_len = weight_len_list[i];
		float *address = (float *)virtual_weight_address;
		int *page_table_addr = (int *)page_table_address;

		end = start + input_len - 1;
		GetWeightKernelLauncher(input, input_len,
			exclusive_weight, num_of_exclusive_weight,
			address, page_table_addr,
			page_size, num_of_weight_page, start, end);
		start = end + 1;
	}

	FreeExclusiveWeight(exclusive_weight);
}

