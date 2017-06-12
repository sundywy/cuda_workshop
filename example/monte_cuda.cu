#include <random>
#include <iostream>

__device__
bool is_inside(float x, float y) 
{
    return sqrt(x*x + y*y) <= 1.0;
}

__global__
void transform(float *xs, float *ys, const int N, int *out)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = id; i < N; i += stride) {
        out[id] = is_inside(xs[id], ys[id]) ? 1 : 0;
    }
}

__global__
void reduce(int *in, int *out)
{
    extern __shared__ int t[];
    int tid = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    t[tid] = in[id];

    __syncthreads();

    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (tid < i) 
          t[tid] += t[tid + i];
        __syncthreads();
    }

    if (tid == 0) 
        out[blockIdx.x] = t[tid];
}

int total_points(float *dx, float *dy, const int N)
{
    const int threads = 512;
    int blocks = (N + threads - 1)/threads;

    int *in, *out, *result, total;

    cudaMalloc(&in, sizeof(int) * N);
    cudaMalloc(&out, sizeof(int) * blocks);
    cudaMalloc(&result, sizeof(int));

    transform<<<blocks, threads>>>(dx, dy, N, in);
    cudaDeviceSynchronize();

    // while (blocks > threads) {
    //     reduce<<<blocks, threads, threads * sizeof(int)>>>(in, out);
    //     cudaDeviceSynchronize();
    //     blocks /= threads;
    //     cudaMemcpy(in, out, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    //     cudaMemset(out, 0, N);
    // }

    reduce<<<blocks, threads, threads * sizeof(int)>>>(in, out);
    cudaDeviceSynchronize();

    reduce<<<1, threads, threads * sizeof(int)>>>(out, result);
    cudaMemcpy(&total, result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
    cudaFree(result);

    return total;
}

int main()
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    const size_t N = 1 << 20;
    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = uniform_dist(e);
        y[i] = uniform_dist(e);
    }

    float *dx, *dy;
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));

    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

    int inside = total_points(dx, dy, N);

    delete[] x;
    delete[] y;

    cudaFree(dx);
    cudaFree(dy);

    float pi = 4.0f * inside / static_cast<float>(N);

    std::cout << pi << std::endl;
}
