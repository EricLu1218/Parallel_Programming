#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

std::runtime_error reprintf(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char *c_str;
    vasprintf(&c_str, fmt, ap);
    va_end(ap);
    std::runtime_error re(c_str);
    free(c_str);
    return re;
}

class Timer
{
    std::chrono::steady_clock::time_point t0;

public:
    Timer() : t0(std::chrono::steady_clock::now()) {}
    ~Timer()
    {
        std::chrono::duration<double> dt = std::chrono::steady_clock::now() - t0;
        printf("validation took %gs\n", dt.count());
    }
};

int main(int argc, char **argv)
{
    Timer t;
    if (argc != 2)
    {
        throw reprintf("must provide exactly 1 argument");
    }
    std::ifstream f(argv[1]);
    if (not f)
    {
        throw reprintf("failed to open file: %s", argv[1]);
    }
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.seekg(0, std::ios_base::end);
    ssize_t fsize = f.tellg();
    if (fsize % sizeof(int))
    {
        throw reprintf("file size %zd is not a multiple of sizeof(int) %zu", fsize, sizeof(int));
    }
    f.seekg(0, std::ios_base::beg);
    int V;
    int E;
    f.read((char *)&V, sizeof V);
    printf("V = %d\n", V);
    if (V < 2 or V > 6000)
    {
        throw reprintf("2 <= V <= 6000 failed");
    }
    f.read((char *)&E, sizeof E);
    printf("E = %d\n", E);
    if (E < 0 or E > V * (V - 1))
    {
        throw reprintf("0 <= E <= V * (V - 1) failed");
    }
    size_t efsize = sizeof(int) * (2 + E * 3);
    if (fsize != efsize)
    {
        throw reprintf("Expected file size %zs but got %zs", efsize, fsize);
    }
    std::vector<int> edges(V * V);
    int exitcode = 0;
    for (int i = 0; i < E; ++i)
    {
        int e[3];
        f.read((char *)e, sizeof e);
        if (e[0] < 0 or e[0] >= V)
        {
            throw reprintf("src[%d] = %d is out of range", i, e[0]);
        }
        if (e[1] < 0 or e[1] >= V)
        {
            throw reprintf("dst[%d] = %d is out of range", i, e[1]);
        }
        if (e[0] == e[1])
        {
            throw reprintf("src[%d] = dst[%d] = %d invalid", i, i, e[0]);
        }
        if (e[2] < 0 or e[2] > 1000)
        {
            throw reprintf("w[%d] = %d is out of range", i, e[2]);
        }
        auto map_key = e[0] * V + e[1];
        if (edges[map_key])
        {
            int j = edges[map_key] - 1;
            throw reprintf("src[%d] = src[%d] = %d and dst[%d] = dst[%d] = %d invalid", j, i, e[0],
                           j, i, e[1]);
        }
        edges[map_key] = i + 1;
    }
    printf("Graph seems OK.\n");
}
