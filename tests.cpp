#define DEBUG_LEVEL 1

#if DEBUG_LEVEL == 1
    #include <string>
#endif

#include "./ml.h"
#include "./tensor.h"
#include "./benchmark.h"

void test_function (float x [4], float* y) 
{
    float sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        sum += (2 * x [i]) + 3;
        y [i] = sum;
    };
};

void run_net ()
{
    // Initialise Network
    size_t dimensions [5] = {4, 10, 50, 10, 4};
    activation_fn functions [4] = {ReLU, ReLU, ReLU, ReLU};
    activation_fn derivatives [4] = {Step, Step, Step, Step};
    float reg_factor = 0.5;
    float learn_rate = 1;
    float learn_rate_time_constant = 300;
    float momentum = 0.5;
    float rms_decay_rate = 0.5;
    int epochs = 10;
    int seed = SEED;

    Network <4> network (
        dimensions, 
        functions, 
        derivatives, 
        Identity, 
        MeanSquaredError, 
        MeanSquaredErrorGradient, 
        reg_factor, 
        learn_rate,
        learn_rate_time_constant,
        momentum,
        rms_decay_rate,
        epochs,
        seed
    );

    // Create Fake Test Data
    size_t size = 10000;
    size_t batch_size = 10;
    float dummy [10000][4];
    float* input [size];
    float* expected [size];

    std::mt19937 generator (SEED);
    std::uniform_real_distribution <float> distribution (0.0, 1.0);

    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < 4; j++)
        {
            dummy [i][j] = distribution (generator);
        };

        input [i] = dummy [i];

        float* y = new float [4];
        test_function (input [i], y);
        expected [i] = y;
    };

    // Train Network
    // float* costs = network.GD_Basic (input, expected, size);
    // float* costs = network.GD_Stochastic (input, expected, size, batch_size);
    // float* costs = network.GD_StochasticMomentum (input, expected, size, batch_size);
    float* costs = network.GD_StochasticNesterov (input, expected, size, batch_size);
    // float* costs = network.GD_RMSProp (input, expected, size, batch_size);
    // float* costs = network.GD_RMSPropNesterov (input, expected, size, batch_size);

    // Process Results
    std::ofstream out;
    out.open ("losses.csv");

    // int num_costs = size;
    int num_costs = size / batch_size;

    for (int i = 0; i < num_costs * epochs; i++) 
    {
        out << costs [i] << ",";
    };
    out.close ();

    system ("python graph.py");
};

void test_tensor ()
{
    size_t dimensions [3] = {2, 3, 2};
    int elements [] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    Tensor <int, 3> T (dimensions, elements);

    // uint indices [] = {1, 2, 0};

    // if (T.index (indices) != T [1][2][0])
    // {
    //     std::cout << T.index (indices) << std::endl;
    //     std::cout << T [1][2][0] << std::endl; 
    // };

    // T.Print ();
    // T.Flip ();
    T.Rotate ();
    T.Print ();
};
    
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

void test_convolve ()
{
    size_t input_dim [3] = {3, 4, 4};
    size_t output_dim [3] = {3, 4, 4};
    size_t kernel_dim [4] = {3, 3, 2, 2};

    std::mt19937 generator (SEED);
    // std::uniform_real_distribution <float> distribution (0.0, 1.0);

    size_t length = 1;
    for (uint i = 0; i < 3; i++)
    {
        length *= input_dim [i];
    };

    std::normal_distribution <float> distribution (0.0, 1.0);

    float input_elements [length];
    for (uint i = 0; i < length; i++)
    {
        // input_elements [i] = distribution (generator);
        input_elements [i] = i;
    };

    length = 1;
    for (uint i = 0; i < 4; i++)
    {
        length *= kernel_dim [i];
    };

    distribution = std::normal_distribution <float> (0.0, 1.0);

    float kernel_elements [length];
    for (uint i = 0; i < length; i++)
    {
        // kernel_elements [i] = distribution (generator);
        kernel_elements [i] = ((i < 4) || (i > 15 && i < 20) || (i > 31)) ? 1 : 0;
    };


    Tensor <float, 3> input (input_dim, input_elements);
    Tensor <float, 4> kernel (kernel_dim, kernel_elements);
    Tensor <float, 3> output (output_dim);

    Convolve <float, 2, true, false> (input, kernel, output, same, 1);

    input.name = "input";
    kernel.name = "kernel";
    output.name = "output";

    input.Print ();
    kernel.Print ();
    output.Print ();


    size_t no_CH_input_dim [3] = {4, 4, 4};
    size_t no_CH_output_dim [3] = {4, 4, 4};
    size_t no_CH_kernel_dim [3] = {2, 2, 2};

    size_t no_CH_length = 1;
    for (uint i = 0; i < 3; i++)
    {
        no_CH_length *= no_CH_input_dim [i];
    };

    float no_CH_input_elements [no_CH_length];
    for (uint i = 0; i < no_CH_length; i++)
    {
        no_CH_input_elements [i] = i;
    };

    no_CH_length = 1;
    for (uint i = 0; i < 3; i++)
    {
        no_CH_length *= no_CH_kernel_dim [i];
    };

    float no_CH_kernel_elements [no_CH_length];
    for (uint i = 0; i < no_CH_length; i++)
    {
        no_CH_kernel_elements [i] = i;
    };

    Tensor <float, 3> no_CH_input (no_CH_input_dim, no_CH_input_elements);
    Tensor <float, 3> no_CH_kernel (no_CH_kernel_dim, no_CH_kernel_elements);
    Tensor <float, 3> no_CH_output (no_CH_output_dim);

    Convolve <float, 3, false, false> (no_CH_input, no_CH_kernel, no_CH_output, same, 1);

    no_CH_input.name = "no_CH_input";
    no_CH_kernel.name = "no_CH_kernel";
    no_CH_output.name = "no_CH_output";

    no_CH_input.Print ();
    no_CH_kernel.Print ();
    no_CH_output.Print ();


    size_t small_input_dim [2] = {4, 4};
    size_t small_output_dim [2] = {4, 4};
    size_t small_kernel_dim [2] = {2, 2};

    size_t small_length = 1;
    for (uint i = 0; i < 2; i++)
    {
        small_length *= small_input_dim [i];
    };

    float small_input_elements [small_length];
    for (uint i = 0; i < small_length; i++)
    {
        small_input_elements [i] = i;
    };

    small_length = 1;
    for (uint i = 0; i < 2; i++)
    {
        small_length *= small_kernel_dim [i];
    };

    float small_kernel_elements [small_length];
    for (uint i = 0; i < small_length; i++)
    {
        small_kernel_elements [i] = 1;
    };

    Tensor <float, 2> small_input (small_input_dim, small_input_elements);
    Tensor <float, 2> small_kernel (small_kernel_dim, small_kernel_elements);
    Tensor <float, 2> small_output (small_output_dim);

    Convolve <float, 2, false, false> (small_input, small_kernel, small_output, same, 1);

    small_input.name = "small_input";
    small_kernel.name = "small_kernel";
    small_output.name = "small_output";

    small_input.Print ();
    small_kernel.Print ();
    small_output.Print ();

    size_t big_input_dim [5] = {3, 2, 2, 3, 2};
    size_t big_output_dim [5] = {3, 2, 2, 3, 2};
    size_t big_kernel_dim [5] = {2, 2, 2, 2, 2};

    size_t big_length = 1;
    for (uint i = 0; i < 5; i++)
    {
        big_length *= big_input_dim [i];
    };

    float big_input_elements [big_length];
    for (uint i = 0; i < big_length; i++)
    {
        big_input_elements [i] = i;
    };

    big_length = 1;
    for (uint i = 0; i < 5; i++)
    {
        big_length *= big_kernel_dim [i];
    };

    float big_kernel_elements [big_length];
    for (uint i = 0; i < big_length; i++)
    {
        big_kernel_elements [i] = 1;
    };

    Tensor <float, 5> big_input (big_input_dim, big_input_elements);
    Tensor <float, 5> big_kernel (big_kernel_dim, big_kernel_elements);
    Tensor <float, 5> big_output (big_output_dim);

    Convolve <float, 5, false, false> (big_input, big_kernel, big_output, same, 1);

    big_input.name = "big_input";
    big_kernel.name = "big_kernel";
    big_output.name = "big_output";

    big_input.Print ();
    big_kernel.Print ();
    big_output.Print ();

    size_t bigger_input_dim [6] = {2, 3, 2, 3, 2, 2};
    size_t bigger_output_dim [6] = {2, 3, 2, 3, 2, 2};
    size_t bigger_kernel_dim [6] = {2, 2, 2, 2, 2, 2};

    size_t bigger_length = 1;
    for (uint i = 0; i < 6; i++)
    {
        bigger_length *= bigger_input_dim [i];
    };

    float bigger_input_elements [bigger_length];
    for (uint i = 0; i < bigger_length; i++)
    {
        bigger_input_elements [i] = i;
    };

    bigger_length = 1;
    for (uint i = 0; i < 6; i++)
    {
        bigger_length *= bigger_kernel_dim [i];
    };

    float bigger_kernel_elements [bigger_length];
    for (uint i = 0; i < bigger_length; i++)
    {
        bigger_kernel_elements [i] = 1;
    };

    Tensor <float, 6> bigger_input (bigger_input_dim, bigger_input_elements);
    Tensor <float, 6> bigger_kernel (bigger_kernel_dim, bigger_kernel_elements);
    Tensor <float, 6> bigger_output (bigger_output_dim);

    Convolve <float, 6, false, false> (bigger_input, bigger_kernel, bigger_output, same, 1);

    bigger_input.name = "bigger_input";
    bigger_kernel.name = "bigger_kernel";
    bigger_output.name = "bigger_output";

    bigger_input.Print ();
    bigger_kernel.Print ();
    bigger_output.Print ();
};

void test_convolution_layer () 
{
    size_t input_dim [2] = {6, 6};
    size_t output_dim [2] = {6, 6};
    size_t kernel_dim [2] = {2, 2};

    size_t size = 1;
    for (uint i = 0; i < 2; i++)
    {
        size *= kernel_dim [i];
    };

    float kernel_elements [size];
    for (uint i = 0; i < size; i++)
    {
        kernel_elements [i] = (i % 2 == 0) ? 2.0 : -1.0;
    };

    Tensor <float, 2> kernel (kernel_dim, kernel_elements);

    std::mt19937 generator (SEED);
    std::uniform_real_distribution <float> distribution (0.0, 1.0);

    size_t length = 1;
    for (uint i = 0; i < 2; i++)
    {
        length *= input_dim [i];
    };

    #define EXAMPLES 10000

    Tensor <float, 2>** input = new Tensor <float, 2>* [EXAMPLES];
    Tensor <float, 2>** expected = new Tensor <float, 2>* [EXAMPLES];

    for (uint i = 0; i < EXAMPLES; i++)
    {
        float input_elements [length];
        for (uint j = 0; j < length; j++)
        {
            input_elements [j] = distribution (generator);
            // input_elements [i] = 1.0;
        };

        input [i] = new Tensor <float, 2> (input_dim, input_elements);
        expected [i] = new Tensor <float, 2> (input_dim);
        Convolve <float, 2, false, false> (*(input [i]), kernel, *(expected [i]));
    };

    Random <1>* r = new Random <1> (16, SEED);
    uint downsampling = 1;

    ConvolutionLayer <float, 2, false> layer (nullptr, input_dim, output_dim, kernel_dim, r, same, downsampling);

    float costs [EXAMPLES];

    for (uint i = 0; i < EXAMPLES; i++)
    {
        // layer.Propagate (input);

        // layer.PrintInput (input);
        // layer.PrintKernel ();
        // layer.PrintOutput ();

        costs [i] = layer.BackPropagate (*(input [i]), *(expected [i]));
        // layer.PrintKernel ();
    };

    layer.PrintKernel ();

    kernel.Print ("actual kernel");

    // Process Results
    std::ofstream out;
    out.open ("losses.csv");

    for (uint i = 0; i < EXAMPLES; i++) 
    {
        out << costs [i] << ",";
    };
    out.close ();

    system ("python graph.py");


    // size_t input_dim [3] = {3, 4, 4};
    // size_t output_dim [3] = {3, 4, 4};
    // size_t kernel_dim [4] = {3, 3, 2, 2};

    // size_t size = 1;
    // for (uint i = 0; i < 4; i++)
    // {
    //     size *= kernel_dim [i];
    // };

    // float kernel_elements [size];
    // for (uint i = 0; i < size; i++)
    // {
    //     kernel_elements [i] = (i % 2 == 0) ? 2.0 : -1.0;
    // };

    // Tensor <float, 4> kernel (kernel_dim, kernel_elements);

    // std::mt19937 generator (SEED);
    // std::uniform_real_distribution <float> distribution (0.0, 1.0);

    // size_t length = 1;
    // for (uint i = 0; i < 3; i++)
    // {
    //     length *= input_dim [i];
    // };

    // #define EXAMPLES 10000

    // Tensor <float, 3>** input = new Tensor <float, 3>* [EXAMPLES];
    // Tensor <float, 3>** expected = new Tensor <float, 3>* [EXAMPLES];

    // for (uint i = 0; i < EXAMPLES; i++)
    // {
    //     float input_elements [length];
    //     for (uint j = 0; j < length; j++)
    //     {
    //         input_elements [j] = distribution (generator);
    //         // input_elements [i] = 1.0;
    //     };

    //     input [i] = new Tensor <float, 3> (input_dim, input_elements);
    //     expected [i] = new Tensor <float, 3> (input_dim);
    //     Convolve <float, 2, true, false> (*(input [i]), kernel, *(expected [i]));
    // };

    // Random <1>* r = new Random <1> (16, SEED);
    // uint downsampling = 1;

    // ConvolutionLayer <float, 2, true> layer (nullptr, input_dim, output_dim, kernel_dim, r, same, downsampling);

    // float costs [EXAMPLES];

    // for (uint i = 0; i < EXAMPLES; i++)
    // {
    //     // layer.Propagate (input);

    //     // layer.PrintInput (input);
    //     // layer.PrintKernel ();
    //     // layer.PrintOutput ();

    //     costs [i] = layer.BackPropagate (*(input [i]), *(expected [i]));
    //     // layer.PrintKernel ();
    // };

    // layer.PrintKernel ();

    // kernel.Print ("actual kernel");

    // // Process Results
    // std::ofstream out;
    // out.open ("losses.csv");

    // for (uint i = 0; i < EXAMPLES; i++) 
    // {
    //     out << costs [i] << ",";
    // };
    // out.close ();

    // system ("python graph.py");
};

void print_prime (uint N)
{
    Profile();

    if (N < 1)
    {
        return;
    };

    uint primes [N];

    primes [0] = 2;
    uint found = 1;

    uint i = 3;

    while (found < N)
    {
        for (uint j = 0; j < found; j++)
        {
            if (i % primes [j] == 0) break;

            if (j == found - 1)
            {
                primes [found] = i;
                found++;
            };
        };

        i++;
    };

    std::cout << primes [N - 1] << std::endl;
};

void print_primes (uint N)
{
    Profile();

    for (uint i = 1; i < N + 1; i++)
    {
        print_prime (i);
    };
};

void test_benchmark ()
{
    Instrumentor::Session ();
    print_primes (100);
};


// ***---------  MAIN  ---------*** //

int main () 
{
    // run_net ();
    // test_tensor ();
    // test_iterate ();
    // test_convolve ();
    // test_convolution_layer ();
    test_benchmark ();
};