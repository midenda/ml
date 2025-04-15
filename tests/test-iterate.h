#include "../tensor.h"

void test (const float x, float y, uint* const index) 
{
    std::cout << "test: " << x << std::endl;
};

template <size_t N>
void test1 (const float in [], float out [], uint* const index) 
{   
    for (int i = 0; i < N; i++)
        std::cout << index [i];
    std::cout << std::endl;
};

void test2 (const Tensor <float, 3>& in, Tensor <float, 3>& out, uint* const index) 
{
    out [index] = in [index];
    out.PrintElements ();
};

void test_iterate ()
{

    // typedef void (*TestFunction) (const float, float, uint* const);

    size_t dimensions [3] = {2, 3, 4};

    float data = 3.1;
    float out = 0.0;

    float data1 [24];
    float out1 [24];
    for (uint i = 0; i < 24; i++)
        data1 [i] = (i % 2 == 1) ? 1.0 : 2.0;

    Tensor <float, 3> data2 (dimensions, data1);
    Tensor <float, 3> out2 (dimensions);

    Iterate <float, float, 3> (test, data, out, dimensions);
    Iterate <float [24], float [24], 3> (test1 <3>, data1, out1, dimensions);
    Iterate <const Tensor <float, 3>&, Tensor <float, 3>&, 3> (test2, data2, out2, dimensions);
};