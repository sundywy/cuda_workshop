# Practical CUDA Programming
Research Computing  
NUS Information Technology  
National University of Singapore



## Please take note
* I'm not an expert in CUDA <!-- .element: class="fragment" -->
* The material for this class is still evolving <!-- .element: class="fragment" -->



# Let's Start



### Comparison between CPU and GPU
![](./resources/cpu_vs_gpu.png)



### Program workflow
![](./resources/program_workflow.png)



# CUDA Runtime API



### Vector addition in CPU
```c++
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
```
```c++
const int N = 1<<20; // 1M elements

// allocate and initialize 2 vectors, e.g. a and b
add(N, x, y);
```



### Vector addition in GPU
``` c++
__global__
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    y[i] = x[i] + y[i];
}
```
```c++
// Allocate memory on GPU
cudaMalloc(&dx, N * sizeof(float));
cudaMalloc(&dy, N * sizeof(float));
  
// Transfer arrays from CPU to GPU
cudaMemcpy(dx, x, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(dy, y, N*sizeof(float), cudaMemcpyHostToDevice);

// Run kernel on 1M elements on the GPU
add<<<1, 1>>>(N, dx, dy);

// Transfer result back to CPU
cudaMemcpy(y, dy, N*sizeof(float), cudaMemcpyDeviceToHost);
```



### Vector addition in GPU

```c++
// Allocate Unified Memory -- accessible from CPU or GPU
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));

// Run kernel on 1M elements on the GPU
add<<<1, 1>>>(N, x, y);

// Wait for GPU to finish before accessing on host
cudaDeviceSynchronize();
cudaFree(x);
cudaFree(y);
```



### Unified Memory</h3>
<img src="./resources/unified_memory.png" style="height: 550px">



### CUDA threads management
<img src="./resources/thread_hierarchy.png" style="height: 550px">



### Vector addition in GPU
```c++
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}
```
```c++
const int nthread = 512, nblock = (N + nthread - 1)/nthread;

dim3 grid(nblock);
dim3 block(nthread);
  
// Run kernel on 1M elements on the GPU
add<<<grid, block>>>(N, x, y);
```



### Vector addition in GPU</h3>
```c++
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x;
  int stride = gridDim.x;

  for (int i = index; i < n; i+= stride)
    y[i] = x[i] + y[i];
}
```

``` c++
const int nblock = 512;
dim3 grid(nblock);

// Run kernel on 1M elements on the GPU
add<<<grid, 1>>>(N, x, y);
```



### Vector addition in GPU
```c++
__global__
void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i+= stride)
    y[i] = x[i] + y[i];
}
```

```c++
const int nthread = 512;
dim3 block(nthread);

// Run kernel on 1M elements on the GPU
add<<<1, block>>>(N, x, y);
```



### CUDA memory hierarchy
<img src="./resources/cuda_memory.png" style="height: 550px">



# Shared memory



### Static shared memory
``` c++
__global__ void reverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```

``` c++
reverse<<<1, n>>>(x, N);
```



### Dynamic shared memory
```c++
__global__ void reverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}
```
```c++
reverse<<<1, n, n*sizeof(int)>>>(x, N);
``` 



### Parallel Reduction
![](./resources/reduction.png)



### Parallel Reduction
```c++
__global__ void reduce(float *in, float *out)
{
  int tid = threadIdx.x;
  int id = blockIdx.x + blockDim.x + threadIdx.x;

  // do reduction
  for (int i = blockDim.x/2; i > 0; i >>= 1) {
    if (tid < i) 
      in[id] += in[id + i];
  
    __syncthreads();
  }

  if (tid == 0) 
      out[blockIdx.x] = in[id];
}
```



### Parallel Reduction
```c++
__global__ void reduce(float *in, float *out)
{
  // load element to shared memory
  extern __shared__ float t[];
  int tid = threadIdx.x;
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  t[tid] = in[id];
  __syncthreads();

  // do reduction
  for (int i = blockDim.x/2; i > 0; i >>= 1) {
    if (tid < i) 
      t[tid] += t[tid + i];
  
    __syncthreads();
  }

  if (tid == 0) 
      out[blockIdx.x] = t[tid];

}
```



### Finite difference methods in CUDA
<img src="./resources/finite_difference.jpg" style="height: 400px">



### Finite difference method in CUDA (naive implementation)
```c++
__global__
void finite_difference(float *in, float* out, int nx, int ny)
{
  __shared__ float s[nx][ny];
  int ni = blockIdx.x * blockDim.x + threadIdx.x;
  int nj = blockIdx.y * blockDim.y + threadIdx.y;

  int dx = gridDim.x * blockDim.x;
  int dy = gridDim.y * blockDim.y;

  for (int i = ni; i < nx; i += dx) {
    for (int j = nj; j < ny; j += dy) {
      int id = i + j * dy;
      s[i][j] = in[id];
    }
    __syncthreads();
  }
      
  for (int i = ni; i < nx; i += dx) {
    for (int j = nj; j < ny; j += dy) {
      s[i][j] = (s[i-1][j] + s[i+1][j] + s[i][j-1] + s[i][j + 1])/4.0;
    }
    __syncthreads();
  }
  // transfer back data to out....
}
```



### Synchronization
```c++ 
// used to synchronize threads in block
__syncthreads()

// used to synchronize all CUDA devices
cudaDeviceSynchronize()
```



### Other important features (some might only available in a newer cards)
* Shuffle functions, e.g. ```__shuffle_up, ___shuffle_xor, etc...```
* Atomic function, e.g. ```atomicAdd, atomicMax, etc...```
* Dynamic parallelism 
* Streams 
* Texture memory



### Example  
### Calculate Pi with Monte Carlo
<img src="./resources/monte_carlo.png" style="height: 450px">



# Thrust Template Library
<https://thrust.github.io>



### What is Thrust?
* C++ STL version of CUDA (general purpose library) <!-- .element: class="fragment" -->
* Work with other hardware (not exclusive to GPU) <!-- .element: class="fragment" -->



# C++ review



### Template
```c++ 
void swap(int& a, int& b) 
{
  int tmp = a;
  b = a;
  a = tmp;
}
``` 
```c++
void swap(std::string& a, std::string& b) 
{
  std::string tmp = a;
  b = a;
  a = tmp;
}
```



### template
```c++ 
int max (int& a, int& b) 
{
  return a > b ? a : b;
}
``` 
```c++
std::string max(std::string& a, std::string& b) 
{
  return a > b ? a : b;
}
```



### template
```c++
template <typename T>
void swap(T& a, T& b) 
{
  T tmp = a;
  b = a;
  a = tmp;
}
```
```c++
template <typename T>
T max(T& a, T& b) 
{
  return a > b ? a : b;
}
```



### iterator
```c++
int sum(int* numbers, const int n)
{
  int total = 0;
  for (int i = 0; i < n; i++>)
    total += numbers[i];

  return total;
}
```



### iterator
```c++
int sum(int* start, int* end)
{
  int total = 0;
  for (int* i = start; i != end; i++>)
    total += *i;

  return total;
}
```



### iterator
![](./resources/iterators-element.png)



### Lambda Function
```c++
std::vector<int> v(10);
std::generate(v.begin(), v.end(), [](){
  return std::rand();
});

int x = 0;
std::generate(v.begin(), v.end(), [&](){
  return x++;
});
```   



### First impression on Thrust
```c++
// sort vector in GPU
int main()
{
  thrust::host_vector<int> h_vec(32 << 20);
  std::generate(h_vec.begin(), h_vec.end(), rand);

  thrust::device_vector<int> d_vec = h_vec;

  thrust::sort(d_vec.begin(), d_vec.end());

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
```



### Thrust Vector
```c++
// create a host_vector with 1024 int
thrust::host_vector<int> h(1024); 

// copy the content of host_vector to device_vector
thrust::device_vector<int> d1 = h; // create an empty device_vector
thrust::device_vector<int> d2(h.begin(), h.end());

// copy back the content of device_vector to host_vector
h = d2
```



### Thrust Vector
```c++
thrust::fill(h.begin(), h.end(), 10); // initialize h with constant 10

thrust::sequence(h.begin(), h.end()); // initialize h with 0, 1, 2, ..., 1023

std::generate(h.begin(), h.end(), [](){
  return std::rand();
}); // generate random number in h

thrust::device_vector<int> d3(1024);

thrust::copy(h.begin(), h.end(), d3.begin());
```



### Thrust Algorithms
```c++
thrust::find(begin, end, value);

thrust::count(begin, end, value);

thrust::merge(begin1, end1, begin2, end2, result);

thrust::sort(begin, end);

thrust::count_if(begin, end, predicate);

// etc...
```



### Thrust Algorithms
```c++
thrust::device_vector<int> in{1,2,3}, out(3);

// compute y = -x
thrust::transform(in.begin(), in.end(), out.begin(), thrust::negate<int>());
```

```c++
thrust::device_vector<int> in1{1,2,3}, in2{4,5,6}

// compute 
thrust::transform(in1.begin(), in1.end(), in2.begin(), out.begin(), [](int x, int y){
  return x + y;
}); // return {5, 7, 9}
```



### Thrust Algorithms
```c++
template<typename T>
T thrust::reduce(begin, end, accumulator, binary_op);
```

```c++  
thrust::host_vector<int> in{1,2,3};

int x = thrust::reduce(in.begin(), in.end(), 0, [](int x, int acc){
  return acc + x;
}); // return 6
```



### Thrust Fancy Iterator
```c++  
thrust::device_vector<int> x{1, 2, 3}, y{10, 20, 30};
```

```c++
auto begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin()));
auto end = thrust::make_zip_iterator(thrust::make_tuple(x.end(), y.end()));

// print (1, 10) (2, 20) (3, 30)
for (auto iter = begin; it != end; it++)
  std::cout << "(" << thrust::get<0>(*iter) << ", " << thrust::get<1>(*iter) << ")";
std::endl;
```



### Functors
```c++
struct Functor(){
  int x;
  Functor(int _x): x(_x){}
  int operator()(){
    return x++;
  }
};
```



### Thrust Functors
```c++
struct Functor(){
  int x 
  Functor(int _x): x(_x);
  __host__ __device__
  int operator()(){
    return x++;
  }
}
```



### Thrust vector to raw pointer
```c++
thrust::device_vector<int> dv(100);

int *raw = thrust::raw_pointer_cast(&dv[0]);

```



### Why no cudaFree?
```c++
template <typename T>
class device_vector {
public: 
  device_vector(size_t n){
    cudaMalloc(&data, n * sizeof(T));
  }
  ~device_vector(){
    cudaFree(data);
  }
private:
  T* data;
};
```



## Pro & Cons
* No threads management <!-- .element: class="fragment" -->
* Many general purpose libraries <!-- .element: class="fragment" -->
* Not enough control on low level memory management <!-- .element: class="fragment" -->



# Example and Exercise



### Other libraries
* cuRand <!-- .element: class="fragment" -->
* CUB <!-- .element: class="fragment" -->
* cuSparse <!-- .element: class="fragment" -->
* ArrayFire <!-- .element: class="fragment" -->
* CUSP <!-- .element: class="fragment" -->
* etc... <!-- .element: class="fragment" -->



### Compilation and job submission
```shell
$ module load gcc-4.8.2
$ module load cuda7.5
$ nvcc monte.cu -o monte -std=c++11
```

```bash
#!/bin/bash
#BSUB -q gpu
#BSUB -o outfile_%J
#BSUB -e errfile_%J

module load gcc-4.8.2
module load cuda7.5

./monte
```

```shell
$ bsub < run.sh
```



# Thank You