#include <boost/sort/spreadsort/float_sort.hpp>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

/* merge to array "a" */
int merge_front(float *a, int a_num, float *b, int b_num, float *t)
{
	int t_num = 0, check = 0;
	float *t_i = t;
	while (a_num--)
	{
		if (!t_num)
		{
			if (b_num && *a > *b)
			{
				check = 1;
				*t_i++ = *a;
				*a = *b++;
				++t_num;
				--b_num;
			}
		}
		else
		{
			if (b_num && *b < *t)
			{
				if (*a > *b)
				{
					check = 1;
					*t_i++ = *a;
					*a = *b++;
					++t_num;
					--b_num;
				}
			}
			else
			{
				if (*a > *t)
				{
					check = 1;
					*t_i++ = *a;
					*a = *t++;
				}
			}
		}
		++a;
	}
	return check;
}

/* merge to array "b" */
int merge_rear(float *a, int a_num, float *b, int b_num, float *t)
{
	int t_num = 0, check = 0;
	float *t_i = t;
	a = a + a_num - 1;
	b = b + b_num - 1;
	while (b_num--)
	{
		if (!t_num)
		{
			if (a_num && *b < *a)
			{
				check = 1;
				*t_i++ = *b;
				*b = *a--;
				++t_num;
				--a_num;
			}
		}
		else
		{
			if (a_num && *a > *t)
			{
				if (*b < *a)
				{
					check = 1;
					*t_i++ = *b;
					*b = *a--;
					++t_num;
					--a_num;
				}
			}
			else
			{
				if (*b < *t)
				{
					check = 1;
					*t_i++ = *b;
					*b = *t++;
				}
			}
		}
		--b;
	}
	return check;
}

int main(int argc, char **argv)
{
	int rank, size, global_data_n, local_data_n, remain_data_n, local_data_start, last_process_id;
	MPI_Group WORLD_GROUP, USED_GROUP;
	MPI_Comm USED_COMM = MPI_COMM_WORLD;
	MPI_File inputFile, outputFile;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* handle arbitrary number of processes */
	global_data_n = atoi(argv[1]);
	if (global_data_n < size)
	{
		MPI_Comm_group(MPI_COMM_WORLD, &WORLD_GROUP);
		int range[1][3] = {{0, global_data_n - 1, 1}};
		MPI_Group_range_incl(WORLD_GROUP, 1, range, &USED_GROUP);
		MPI_Comm_create(MPI_COMM_WORLD, USED_GROUP, &USED_COMM);
		// if task isn't in "USED_GROUP", its "USED_COMM" is "MPI_COMM_NULL".
		if (USED_COMM == MPI_COMM_NULL)
		{
			MPI_Finalize();
			return 0;
		}
		size = global_data_n;
	}
	last_process_id = size - 1;

	/* distribute input data to different process */
	local_data_n = global_data_n / size;
	remain_data_n = global_data_n % size;
	float *recv = new float[local_data_n + 1]; // use to receive
	float *temp = new float[local_data_n + 1]; // use to merge
	if (rank < remain_data_n)
	{
		++local_data_n;
		local_data_start = rank * local_data_n;
	}
	else
	{
		local_data_start = rank * local_data_n + remain_data_n;
	}
	float *data = new float[local_data_n];

	/* calculate how many data change in each phase */
	int recv_before_data_n, recv_next_data_n;
	if (rank + 1 == remain_data_n)
		recv_next_data_n = local_data_n - 1;
	else
		recv_next_data_n = local_data_n;
	if (rank == remain_data_n)
		recv_before_data_n = local_data_n + 1;
	else
		recv_before_data_n = local_data_n;

	/* read data from inputfile */
	MPI_File_open(USED_COMM, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &inputFile);
	MPI_File_read_at(inputFile, sizeof(float) * local_data_start, data, local_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&inputFile);

	/* execute odd-even sort */
	boost::sort::spreadsort::float_sort(data, data + local_data_n);
	int even_check = 0, odd_check = 0, check = 0, terminate = 1;
	while (terminate)
	{
		/* even phase */
		// if rank is even and isn't last process, it needs to send and receive data with (rank + 1).
		if (!(rank & 1) && rank != last_process_id)
		{
			MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank + 1, 0,
						 recv, recv_next_data_n, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);
			even_check = merge_front(data, local_data_n, recv, recv_next_data_n, temp);
		}
		// if rank is odd, it needs to send and receive data with (rank - 1).
		else if (rank & 1)
		{
			MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank - 1, 0,
						 recv, recv_before_data_n, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
			even_check = merge_rear(recv, recv_before_data_n, data, local_data_n, temp);
		}

		/* odd phase */
		// if rank is odd and isn't last process, it needs to send and receive data with (rank + 1).
		if ((rank & 1) && rank != last_process_id)
		{
			MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank + 1, 0,
						 recv, recv_next_data_n, MPI_FLOAT, rank + 1, 0, USED_COMM, MPI_STATUS_IGNORE);
			odd_check = merge_front(data, local_data_n, recv, recv_next_data_n, temp);
		}
		// if rank is even and isn't first process, it needs to send and receive data with (rank - 1).
		else if (!(rank & 1) && rank != 0)
		{
			MPI_Sendrecv(data, local_data_n, MPI_FLOAT, rank - 1, 0,
						 recv, recv_before_data_n, MPI_FLOAT, rank - 1, 0, USED_COMM, MPI_STATUS_IGNORE);
			odd_check = merge_rear(recv, recv_before_data_n, data, local_data_n, temp);
		}
		check = odd_check | even_check;
		MPI_Allreduce(&check, &terminate, 1, MPI_INT, MPI_SUM, USED_COMM);
	}

	/* write data to outputfile */
	MPI_File_open(USED_COMM, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outputFile);
	MPI_File_write_at(outputFile, sizeof(float) * local_data_start, data, local_data_n, MPI_FLOAT, MPI_STATUS_IGNORE);
	MPI_File_close(&outputFile);

	delete data, recv;
	MPI_Finalize();
	return 0;
}