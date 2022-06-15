/* without vectorization */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write_png(const char *filename, const int iters, const int width, const int height, const int *buffer, const int size, const int partition)
{
	FILE *fp = fopen(filename, "wb");
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
	png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);
	size_t row_size = 3 * width * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_size);
	int y = 0;
	for (int k = 0; k < height; ++k)
	{
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x)
		{
			int p = buffer[y * width + x];
			png_bytep color = row + x * 3;
			if (p != iters)
			{
				if (p & 16)
				{
					color[0] = 240;
					color[1] = color[2] = (p & 15) << 4;
				}
				else
					color[0] = (p & 15) << 4;
			}
		}
		png_write_row(png_ptr, row);
		y += partition;
		if (y >= partition * size)
			y = y % partition + 1;
	}
	free(row);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
}

int main(int argc, char **argv)
{
	/* initial MPI*/
	int rank, size;
	MPI_Group WORLD_GROUP, USED_GROUP;
	MPI_Comm USED_COMM = MPI_COMM_WORLD;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* argument parsing */
	assert(argc == 9);
	const char *filename = argv[1];
	const int iters = strtol(argv[2], 0, 10);
	const double left = strtod(argv[3], 0);
	const double right = strtod(argv[4], 0);
	const double lower = strtod(argv[5], 0);
	const double upper = strtod(argv[6], 0);
	const int width = strtol(argv[7], 0, 10);
	const int height = strtol(argv[8], 0, 10);

	/* handle arbitrary number of processes */
	if (height < size)
	{
		MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
		int range[1][3] = {{0, height - 1, 1}};
		MPI_Group_range_incl(WORLD_GROUP, 1, range, &USED_GROUP);
		MPI_Comm_create(MPI_COMM_WORLD, USED_GROUP, &USED_COMM);
		if (USED_COMM == MPI_COMM_NULL)
		{
			MPI_Finalize();
			return 0;
		}
		size = height;
	}

	/* mandelbrot set */
	const int partition = ceil((double)height / size);
	const double x_offset = (right - left) / width, y_offset = (upper - lower) / height;
	int *pixel = (int *)malloc(partition * width * sizeof(int)), row_index = 0;
	for (int j = height - 1 - rank; j >= 0; j -= size)
	{
		double y0 = j * y_offset + lower;
#pragma omp parallel for schedule(dynamic, 1)
		for (int i = 0; i < width; ++i)
		{
			double x0 = i * x_offset + left;
			double x = 0, y = 0;
			double length_squared = 0;
			int repeats = 0;
			while (repeats < iters && length_squared < 4)
			{
				double temp = x * x - y * y + x0;
				y = 2 * x * y + y0;
				x = temp;
				length_squared = x * x + y * y;
				++repeats;
			}
			pixel[row_index * width + i] = repeats;
		}
		++row_index;
	}

	/* allocate memory for image */
	int *image = (int *)malloc(size * partition * width * sizeof(int));
	MPI_Gather(pixel, partition * width, MPI_INT, image, partition * width, MPI_INT, 0, USED_COMM);

	/* draw and cleanup */
	if (rank == 0)
		write_png(filename, iters, width, height, image, size, partition);
	free(pixel);
	free(image);
	MPI_Finalize();
	return 0;
}
