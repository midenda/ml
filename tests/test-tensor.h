#define DEBUG_LEVEL 1
#define PROFILING 1
#define SEED 999

#if DEBUG_LEVEL == 1
    #include <string>
#endif

#include "../tensor.h"
#include "../benchmark.h"

void test_size ()
{
    size_t input_dim [4] = {3, 5, 7, 11};
    Tensor <float, 4> T (input_dim);
    T.Randomise ();

    // std::cout << "Size: " << T.size << std::endl;

    // size_t expected = 3 * 5 * 7 * sizeof (Tensor <float, 1>) 
    //                 + 3 * 5 * sizeof (Tensor <float, 2>) 
    //                 + 3 * sizeof (Tensor <float, 3>) 
    //                 + sizeof (T)
    //                 + T.length * sizeof (float);

    // std::cout << "Expected size: " << expected << std::endl;

    size_t overhead = T.size - T.length * sizeof (float);
    float ratio = (float)overhead / (float)(T.size);

    // std::cout << "Size - sizeof (elements): " << overhead << std::endl;
    std::cout << "Ratio: " << ratio << std::endl; 

};

void test_tensor ()
{
    size_t dimensions [3] = {2, 3, 2};
    int elements [] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor <int, 3> T (dimensions, elements);

    uint indices [] = {1, 2, 0};

    if (T.index (indices) != T [1][2][0])
    {
        std::cout << T.index (indices) << std::endl;
        std::cout << T [1][2][0] << std::endl; 
    };

    
    T.Print ();
    // T.Flip ();
    // T.Rotate ();
    // T.Print ();

    test_size ();
};