#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int INF = (1 << 30) - 1;
int vertex_num, edge_num, matrix_size;
int *dist;

double cal_time(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec - start.tv_nsec) < 0)
	{
		temp.tv_sec = end.tv_sec - start.tv_sec - 1;
		temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
	}
	else
	{
		temp.tv_sec = end.tv_sec - start.tv_sec;
		temp.tv_nsec = end.tv_nsec - start.tv_nsec;
	}
	return temp.tv_sec + (double)temp.tv_nsec / 1000000000.0;
}

__device__ __host__ size_t index_convert(int i, int j, int row_size)
{
	return i * row_size + j;
}

void input(char *input_file_path, int &block_factor)
{
	FILE *input_file = fopen(input_file_path, "rb");
	fread(&vertex_num, sizeof(int), 1, input_file);
	fread(&edge_num, sizeof(int), 1, input_file);

	matrix_size = ceil((double)vertex_num / (double)block_factor) * block_factor;
	cudaMallocHost((void **)&dist, matrix_size * matrix_size * sizeof(int));
	for (int i = 0; i < matrix_size; ++i)
	{
		for (int j = 0; j < matrix_size; ++j)
		{
			if (i != j)
				dist[index_convert(i, j, matrix_size)] = INF;
			else if (i < vertex_num)
				dist[index_convert(i, j, matrix_size)] = 0;
			else
				dist[index_convert(i, j, matrix_size)] = INF;
		}
	}

	int data[3];
	for (int i = 0; i < edge_num; ++i)
	{
		fread(data, sizeof(int), 3, input_file);
		dist[index_convert(data[0], data[1], matrix_size)] = data[2];
	}
	fclose(input_file);
}

void output(char *output_file_path)
{
	FILE *output_file = fopen(output_file_path, "w");
	for (int i = 0; i < vertex_num; ++i)
	{
		fwrite(&dist[index_convert(i, 0, matrix_size)], sizeof(int), vertex_num, output_file);
	}
	fclose(output_file);
}

__constant__ int size[3]; //matrix size, block_factor, grid_size

__global__ void phase1(int *d_dist, int round)
{
	__shared__ int share[4 * 1024];
	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset = size[1] * round;
	int j_offset = size[1] * round;

	share[index_convert(j, i, size[1])] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
#pragma unroll 32
	for (int k = 0; k < size[1]; ++k)
	{
		__syncthreads();
		if (share[index_convert(j, i, size[1])] > share[index_convert(j, k, size[1])] + share[index_convert(k, i, size[1])])
			share[index_convert(j, i, size[1])] = share[index_convert(j, k, size[1])] + share[index_convert(k, i, size[1])];
	}
	d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = share[index_convert(j, i, size[1])];
}

__global__ void phase2(int *d_dist, int round)
{
	__shared__ int share[3 * 4 * 1024];
	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset, j_offset;
	if (blockIdx.x == 0)
	{
		i_offset = size[1] * ((round + blockIdx.y + 1) % size[2]);
		j_offset = size[1] * round;
		share[index_convert(i, j, size[1])] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
		share[index_convert(i + size[1], j, size[1])] = share[index_convert(i, j, size[1])];
		share[index_convert(i + 2 * size[1], j, size[1])] = d_dist[index_convert(j_offset + i, j_offset + j, size[0])];
	}
	else
	{
		i_offset = size[1] * round;
		j_offset = size[1] * ((round + blockIdx.y + 1) % size[2]);
		share[index_convert(i, j, size[1])] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
		share[index_convert(i + size[1], j, size[1])] = d_dist[index_convert(i_offset + i, i_offset + j, size[0])];
		share[index_convert(i + 2 * size[1], j, size[1])] = share[index_convert(i, j, size[1])];
	}

#pragma unroll 32
	for (int k = 0; k < size[1]; ++k)
	{
		__syncthreads();
		if (share[index_convert(i, j, size[1])] >
			share[index_convert(i + size[1], k, size[1])] + share[index_convert(k + 2 * size[1], j, size[1])])
			share[index_convert(i, j, size[1])] =
				share[index_convert(i + size[1], k, size[1])] + share[index_convert(k + 2 * size[1], j, size[1])];
	}
	d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = share[index_convert(i, j, size[1])];
}

__global__ void phase3(int *d_dist, int round)
{
	__shared__ int share[3 * 4 * 1024];
	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset = size[1] * ((round + blockIdx.y + 1) % size[2]);
	int j_offset = size[1] * ((round + blockIdx.x + 1) % size[2]);
	int r_offset = size[1] * round;

	share[index_convert(i, j, size[1])] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
	share[index_convert(i + size[1], j, size[1])] = d_dist[index_convert(i_offset + i, r_offset + j, size[0])];
	share[index_convert(i + 2 * size[1], j, size[1])] = d_dist[index_convert(r_offset + i, j_offset + j, size[0])];
#pragma unroll 32
	for (int k = 0; k < size[1]; ++k)
	{
		__syncthreads();
		if (share[index_convert(i, j, size[1])] >
			share[index_convert(i + size[1], k, size[1])] + share[index_convert(k + 2 * size[1], j, size[1])])
			share[index_convert(i, j, size[1])] =
				share[index_convert(i + size[1], k, size[1])] + share[index_convert(k + 2 * size[1], j, size[1])];
	}
	d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = share[index_convert(i, j, size[1])];
}

int main(int argc, char **argv)
{
	double total_time, bfd_time;
	timespec total_time1, total_time2, bfd_time1, bfd_time2;

	clock_gettime(CLOCK_MONOTONIC, &total_time1);
	cudaSetDevice(0);
	int block_factor = 32;
	if (argc == 4)
		block_factor = atoi(argv[3]);
	input(argv[1], block_factor);
	int grid_size = matrix_size / block_factor;

	int size_info[3] = {matrix_size, block_factor, grid_size};
	cudaMemcpyToSymbol(size, size_info, 3 * sizeof(int));

	int *d_dist;
	clock_gettime(CLOCK_MONOTONIC, &bfd_time1);
	cudaMalloc(&d_dist, (size_t)sizeof(int) * matrix_size * matrix_size);
	cudaMemcpy(d_dist, dist, (size_t)sizeof(int) * matrix_size * matrix_size, cudaMemcpyHostToDevice);
	dim3 block(block_factor, block_factor);
	dim3 grid2(2, grid_size - 1);
	dim3 grid3(grid_size - 1, grid_size - 1);
	for (int r = 0; r < grid_size; ++r)
	{
		phase1<<<1, block>>>(d_dist, r);
		phase2<<<grid2, block>>>(d_dist, r);
		phase3<<<grid3, block>>>(d_dist, r);
	}
	cudaMemcpy(dist, d_dist, (size_t)sizeof(int) * matrix_size * matrix_size, cudaMemcpyDeviceToHost);
	clock_gettime(CLOCK_MONOTONIC, &bfd_time2);

	output(argv[2]);
	cudaFree(d_dist);
	cudaFree(dist);

	clock_gettime(CLOCK_MONOTONIC, &total_time2);
	bfd_time = cal_time(bfd_time1, bfd_time2);
	total_time = cal_time(total_time1, total_time2);
	printf(" vertex:   %d\n", vertex_num);
	printf(" I/O time: %.5f\n", total_time - bfd_time);
	printf(" cal time: %.5f\n", bfd_time);
	printf(" runtime:  %.5f\n", total_time);
	return 0;
}