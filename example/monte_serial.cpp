#include <random>
#include <iostream>

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
    float *x = new float[N];
    float *y = new float[N];

    int inside = 0;
    for (int i = 0; i < N; i++) {
        x[i] = uniform_dist(e);
        y[i] = uniform_dist(e);

        if (is_inside(x[i], y[i]))
            inside++;
    }

    delete[] x;
    delete[] y;

    auto pi = 4.0f * inside / static_cast<float>(N);

    std::cout << pi << std::endl;
}
