/* without vectorization */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int thread_num, iters, width, height, *image;
double left, right, lower, upper;

void *cal(void *threadid)
{
	int tid = *((int *)threadid);
	for (int j = 0; j < height; ++j)
	{
		double y0 = j * ((upper - lower) / height) + lower;
		for (int i = tid; i < width; i += thread_num)
		{
			double x0 = i * ((right - left) / width) + left;
			double x = 0;
			double y = 0;
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
			image[j * width + i] = repeats;
		}
	}
}

void write_png(const char *filename, const int *buffer)
{
	FILE *fp = fopen(filename, "wb");
	assert(fp);
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	assert(png_ptr);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	assert(info_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
	png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);
	size_t row_size = 3 * width * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_size);
	for (int y = 0; y < height; ++y)
	{
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x)
		{
			int p = buffer[(height - 1 - y) * width + x];
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
	}
	free(row);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
}

int main(int argc, char **argv)
{
	/* detect how many CPUs are available */
	cpu_set_t cpu_set;
	sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
	thread_num = CPU_COUNT(&cpu_set);
	int tid[thread_num];
	pthread_t thread[thread_num];

	/* argument parsing */
	assert(argc == 9);
	const char *filename = argv[1];
	iters = strtol(argv[2], 0, 10);
	left = strtod(argv[3], 0);
	right = strtod(argv[4], 0);
	lower = strtod(argv[5], 0);
	upper = strtod(argv[6], 0);
	width = strtol(argv[7], 0, 10);
	height = strtol(argv[8], 0, 10);

	/* allocate memory for image */
	image = (int *)malloc(width * height * sizeof(int));
	assert(image);

	/* mandelbrot set */
	for (int i = 0; i < thread_num; ++i)
	{
		tid[i] = i;
		pthread_create(&thread[i], NULL, cal, (void *)&tid[i]);
	}
	for (int i = 0; i < thread_num; ++i)
		pthread_join(thread[i], NULL);

	/* draw and cleanup */
	write_png(filename, image);
	free(image);
	pthread_exit(NULL);
	return 0;
}
