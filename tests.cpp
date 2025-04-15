#define DEBUG_LEVEL 1
#define PROFILING 1
#define SEED 999
// #define EXAMPLES 12000
#define EXAMPLES 100000

#if DEBUG_LEVEL == 1
    #include <string>
#endif

#include "./ml.h"
#include "./tensor.h"
#include "./benchmark.h"
#include "./regression.h"

#include "./tests/test-benchmark.h"
#include "./tests/test-convolution-layer.h"
#include "./tests/test-convolve.h"
#include "./tests/test-iterate.h"
#include "./tests/test-network.h"
#include "./tests/test-recurrent-layer.h"
#include "./tests/test-regression.h"
#include "./tests/test-tensor.h"

// ***---------  MAIN  ---------*** //

int main () 
{
    test_size ();
    test_benchmark ();
    test_tensor ();
    test_iterate ();
    test_regression ();
    test_convolve ();
    // test_convolution_layer (); // Seg fault
    test_recurrent_layer ();
    run_net ();
};