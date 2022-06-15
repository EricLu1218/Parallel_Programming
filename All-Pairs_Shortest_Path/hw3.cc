#include <fstream>
#include <iostream>
#include <omp.h>
#include <vector>

#define INF 1073741823

using namespace std;

int main(int argc, char **argv)
{
	double input_time, output_time, compute_time, total_time;
	if (argc != 3)
		return 0;

	ifstream input_file(argv[1], ios::in | ios::binary);
	ofstream output_file(argv[2], ios::out | ios::binary);
	ios::sync_with_stdio(false);

	int vertex_num, edge_num;
	input_file.read((char *)&vertex_num, 4);
	input_file.read((char *)&edge_num, 4);

	total_time = omp_get_wtime();
	vector<vector<int>> dist;
	dist.resize(vertex_num);
#pragma omp parallel for schedule(guided, 1)
	for (int i = 0; i < vertex_num; ++i)
	{
		dist[i].resize(vertex_num, INF);
		dist[i][i] = 0;
	}

	input_time = omp_get_wtime();
	int source, destination, weight;
	while (input_file.read((char *)&source, 4))
	{
		input_file.read((char *)&destination, 4);
		input_file.read((char *)&weight, 4);
		dist[source][destination] = weight;
	}
	input_time = omp_get_wtime() - input_time;

	compute_time = omp_get_wtime();
	for (int k = 0; k < vertex_num; ++k)
	{
#pragma omp parallel for schedule(guided, 1) collapse(2)
		for (int i = 0; i < vertex_num; ++i)
		{
			for (int j = 0; j < vertex_num; ++j)
			{
				if (dist[i][j] > dist[i][k] + dist[k][j] && dist[i][k] != INF)
					dist[i][j] = dist[i][k] + dist[k][j];
			}
		}
	}
	compute_time = omp_get_wtime() - compute_time;

	output_time = omp_get_wtime();
	for (int i = 0; i < vertex_num; ++i)
	{
		for (int j = 0; j < vertex_num; ++j)
		{
			output_file.write((char *)&dist[i][j], 4);
		}
	}
	output_time = omp_get_wtime() - output_time;
	total_time = omp_get_wtime() - total_time;

	cout << input_time + output_time << " " << compute_time << " " << total_time << endl;
	return 0;
}