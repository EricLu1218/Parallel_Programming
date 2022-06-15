#include <math.h>
#include <omp.h>
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

void input(char *input_file_path, int block_factor)
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
	__shared__ int pivot[1024];
	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset = 32 * round;
	int j_offset = 32 * round;

	pivot[index_convert(i, j, 32)] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
#pragma unroll 32
	for (int k = 0; k < 32; ++k)
	{
		__syncthreads();
		if (pivot[index_convert(i, j, 32)] > pivot[index_convert(i, k, 32)] + pivot[index_convert(k, j, 32)])
			pivot[index_convert(i, j, 32)] = pivot[index_convert(i, k, 32)] + pivot[index_convert(k, j, 32)];
	}
	d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = pivot[index_convert(i, j, 32)];
}

__global__ void phase2(int *d_dist, int round)
{
	__shared__ int self[1024], pivot[1024];
	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset, j_offset;
	if (blockIdx.x == 0 && blockIdx.y != round)
	{
		i_offset = 32 * blockIdx.y;
		j_offset = 32 * round;

		self[index_convert(i, j, 32)] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
		pivot[index_convert(i, j, 32)] = d_dist[index_convert(j_offset + i, j_offset + j, size[0])];
#pragma unroll 32
		for (int k = 0; k < 32; ++k)
		{
			__syncthreads();
			if (self[index_convert(i, j, 32)] > self[index_convert(i, k, 32)] + pivot[index_convert(k, j, 32)])
				self[index_convert(i, j, 32)] = self[index_convert(i, k, 32)] + pivot[index_convert(k, j, 32)];
		}
		d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = self[index_convert(i, j, 32)];
	}
	else if (blockIdx.y != round)
	{
		i_offset = 32 * round;
		j_offset = 32 * blockIdx.y;

		self[index_convert(i, j, 32)] = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
		pivot[index_convert(i, j, 32)] = d_dist[index_convert(i_offset + i, i_offset + j, size[0])];
#pragma unroll 32
		for (int k = 0; k < 32; ++k)
		{
			__syncthreads();
			if (self[index_convert(i, j, 32)] > pivot[index_convert(i, k, 32)] + self[index_convert(k, j, 32)])
				self[index_convert(i, j, 32)] = pivot[index_convert(i, k, 32)] + self[index_convert(k, j, 32)];
		}
		d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = self[index_convert(i, j, 32)];
	}
}

__global__ void phase3(int *d_dist, int round, int grid_offset)
{
	__shared__ int col[1024], row[1024];
	int self;

	int block_i = grid_offset + blockIdx.y;
	int block_j = blockIdx.x;
	if (block_i == round || block_j == round)
		return;

	int i = threadIdx.y;
	int j = threadIdx.x;

	int i_offset = 32 * block_i;
	int j_offset = 32 * block_j;
	int r_offset = 32 * round;

	self = d_dist[index_convert(i_offset + i, j_offset + j, size[0])];
	col[index_convert(i, j, 32)] = d_dist[index_convert(i_offset + i, r_offset + j, size[0])];
	row[index_convert(i, j, 32)] = d_dist[index_convert(r_offset + i, j_offset + j, size[0])];

#pragma unroll 32
	for (int k = 0; k < 32; ++k)
	{
		__syncthreads();
		if (self > col[index_convert(i, k, 32)] + row[index_convert(k, j, 32)])
			self = col[index_convert(i, k, 32)] + row[index_convert(k, j, 32)];
	}
	d_dist[index_convert(i_offset + i, j_offset + j, size[0])] = self;
}

int main(int argc, char **argv)
{
	const int block_factor = 32, device_num = 2;
	input(argv[1], block_factor);
	int grid_size = matrix_size / block_factor;

	int *d_dist[2];
#pragma omp parallel num_threads(device_num)
	{
		int device_id = omp_get_thread_num();
		cudaSetDevice(device_id);

		int size_info[3] = {matrix_size, block_factor, grid_size};
		cudaMemcpyToSymbol(size, size_info, 3 * sizeof(int));

		int grid_partition = grid_size / device_num;
		int grid_offset = device_id * grid_partition;
		int grid_count = grid_partition;
		if (device_id == device_num - 1)
			grid_count += grid_size % device_num;
		size_t grid_start = grid_offset * block_factor * matrix_size;

		cudaMalloc(&(d_dist[device_id]), (size_t)sizeof(int) * matrix_size * matrix_size);
#pragma omp barrier
		cudaMemcpy(&(d_dist[device_id][grid_start]), &(dist[grid_start]),
				   (size_t)sizeof(int) * block_factor * grid_count * matrix_size, cudaMemcpyHostToDevice);
		dim3 block(block_factor, block_factor);
		dim3 grid2(2, grid_size);
		dim3 grid3(grid_size, grid_count);
		for (int r = 0; r < grid_size; ++r)
		{
			if (grid_offset <= r && r < grid_offset + grid_count)
			{
				size_t copy_start = r * block_factor * matrix_size;
				if (device_id == 0)
					cudaMemcpy(&(d_dist[1][copy_start]), &(d_dist[0][copy_start]),
							   (size_t)sizeof(int) * block_factor * matrix_size, cudaMemcpyDeviceToDevice);
				else
					cudaMemcpy(&(d_dist[0][copy_start]), &(d_dist[1][copy_start]),
							   (size_t)sizeof(int) * block_factor * matrix_size, cudaMemcpyDeviceToDevice);
			}
#pragma omp barrier
			phase1<<<1, block>>>(d_dist[device_id], r);
			phase2<<<grid2, block>>>(d_dist[device_id], r);
			phase3<<<grid3, block>>>(d_dist[device_id], r, grid_offset);
		}
		cudaMemcpy(&(dist[grid_start]), &(d_dist[device_id][grid_start]),
				   (size_t)sizeof(int) * block_factor * grid_count * matrix_size, cudaMemcpyDeviceToHost);
		cudaFree(d_dist[omp_get_thread_num()]);
#pragma omp barrier
	}

	output(argv[2]);
	cudaFree(dist);
	return 0;
}