#include <random>
#include <iostream>
#include <vector>
#include <algorithm>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
// #include <utility>

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
    std::vector<float> x(N), y(N);

    std::generate(x.begin(), x.end(), [&](){
        return uniform_dist(e);
    });

    std::generate(y.begin(), y.end(), [&](){
        return uniform_dist(e);
    });

    auto begin = boost::make_zip_iterator(boost::make_tuple(x.begin(), y.begin()));
    auto end = boost::make_zip_iterator(boost::make_tuple(x.end(), y.end()));

    size_t inside = std::count_if(begin, end, [](auto t){
        return is_inside(boost::get<0>(t), boost::get<1>(t));
    });

    auto pi = 4.0f * inside / static_cast<float>(N);

    std::cout << pi << std::endl;
}
