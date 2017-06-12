#include <random>
#include <iostream>

#include <thrust/host_vector.h>

__device__
bool is_inside(float x, float y) 
{
    return sqrt(x*x + y*y) <= 1.0;
}

int main()
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    const size_t N = 1 << 20;

    thrust::host_vector<float> x(N), y(N);
    thrust::generate(x.begin(), x.end(), [&](){
        return uniform_dist(e);
    });

    thrust::generate(y.begin(), y.end(), [&](){
        return uniform_dist(e);
    });

    // Your code here...
    // use thrust::count_if, thrust::zip_iterator, thrust::tuple

}
