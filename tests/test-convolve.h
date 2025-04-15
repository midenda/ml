#include "../ml.h"

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