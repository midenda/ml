#include "../ml.h"

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

    Tensor <float, 2>** input = new Tensor <float, 2>* [EXAMPLES];
    Tensor <float, 2>** expected = new Tensor <float, 2>* [EXAMPLES];

    for (uint i = 0; i < EXAMPLES; i++)
    {
        float input_elements [length];
        float expected_elements [length];
        for (uint j = 0; j < length; j++)
        {
            input_elements [j] = distribution (generator);
            // input_elements [i] = 1.0;
            expected_elements [i] = 1.0;
        };

        input [i] = new Tensor <float, 2> (input_dim, input_elements);
        expected [i] = new Tensor <float, 2> (input_dim, expected_elements);
        Convolve <float, 2, false, false> (*(input [i]), kernel, *(expected [i]));
    };

    NormalisedRandom <1>* r = new NormalisedRandom <1> (16, SEED);
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

    system ("python3 graph.py losses.csv");


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

    // NormalisedRandom <1>* r = new NormalisedRandom <1> (16, SEED);
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

    // system ("python3 graph.py losses.csv");
};