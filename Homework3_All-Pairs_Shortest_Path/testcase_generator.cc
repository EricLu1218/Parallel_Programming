#include <fstream>
#include <iostream>
#include <vector>

#define INF 1073741823

using namespace std;

int main(int argc, char **argv)
{
	ofstream output_file("testcase.in", ios::out | ios::binary);
	ios::sync_with_stdio(false);

	int vertex_num = 6000, edge_num = vertex_num * (vertex_num - 1);

	vector<vector<int>> weight;
	weight.resize(vertex_num);
	for (int i = 0; i < vertex_num; ++i)
	{
		weight[i].resize(vertex_num, INF);
	}

	int cnt = 0;
	for (int i = 0; i < vertex_num; ++i)
	{
		for (int j = 0; j < vertex_num; ++j)
		{
			weight[i][j] = cnt++ % 1000;
		}
	}

	output_file.write((char *)&vertex_num, 4);
	output_file.write((char *)&edge_num, 4);
	for (int i = 0; i < vertex_num; ++i)
	{
		for (int j = 0; j < vertex_num; ++j)
		{
			if (i == j)
				continue;
			output_file.write((char *)&i, 4);
			output_file.write((char *)&j, 4);
			output_file.write((char *)&weight[i][j], 4);
		}
	}
	return 0;
}