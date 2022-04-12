#define DEBUG_LEVEL 1
#define SEED 999

#include <cmath>
#include <iostream>
#include <fstream>
#include <random> 
#include <functional>
#include <algorithm> // std::shuffle

#if DEBUG_LEVEL == 1
    #include <string>
#endif

#include "./tensor.h"

// Implementation of std::conditional
template <bool, typename T, typename F>
struct conditional
{
    typedef T type;
};

template <typename T, typename F>
struct conditional <false, T, F>
{
    typedef F type;
};


template <typename FunctionType, typename InputType, typename OutputType, size_t N>
void Iterate (FunctionType f, InputType x, OutputType y, size_t dimensions [N], Interim interim);

template <typename FunctionType, typename InputType, typename OutputType, size_t M, size_t N>
void Iterate (FunctionType f, InputType x, OutputType y, size_t dimensions [M], uint outer_index [N], Interim interim);


enum ConvolutionType { valid, optimal, same, full };

typedef float (*activation_fn) (float);
typedef float* (*output_fn) (float[], size_t);
typedef float (*loss_fn) (float[], float[], size_t);
typedef float* (*loss_gradient) (float[], float[], size_t);

template <size_t Dim, bool Chns, bool Backprop>
struct ConvolutionInput
{
    const Tensor <float, Dim + Chns>& input;
    const Tensor <float, Dim + ((2 - Backprop) * Chns)>& kernel;

    uint downsample;
    uint padding [Dim];

    ConvolutionInput 
    (
        const Tensor <float, Dim + Chns>& input, 
        const Tensor <float, Dim + ((2 - Backprop) * Chns)>& kernel, 
        ConvolutionType type = same, 
        uint downsample = 1
    )
        : input {input}, kernel {kernel}, downsample {downsample}
    {
        /* 
        m: input width
        k: kernel width

        if no zero padding (VALID convolution):
            output has width                           m - k + 1 

        if zero padding (SAME convolution):
            enough zeroes added to maintain size
            output width = input width                 m

        if zero padding (FULL convolution):
            enough zeroes added so that each input pixel is visited k times
            output width                               m + k - 1

        */

        switch (type)
        {
            case valid:    
                for (uint i = 0; i < Dim; i++)
                {   
                    padding [i] = 0;
                };   
                break;

            case optimal:
                for (uint i = 0; i < Dim; i++)
                {   
                    padding [i] = kernel.dimensions [i + ((2 - Backprop) * Chns)] / 3;  
                };   
                break;

            case same:        
                for (uint i = 0; i < Dim; i++)
                {   
                    padding [i] = kernel.dimensions [i + ((2 - Backprop) * Chns)] / 2;  
                };   
                break;

            case full:        
                for (uint i = 0; i < Dim; i++)
                {   
                    padding [i] = kernel.dimensions [i + ((2 - Backprop) * Chns)] - 1;  
                };
                break;
        };
    };
};

template <size_t Dim, bool Chns, bool Backprop>
struct ConvolutionIterFunc
{
    typedef void (*Inner)    (const ConvolutionInput <Dim, Chns, Backprop>&, Tensor <float, Dim + ((1 + Backprop) * Chns)>&, uint [2 * (Dim + Chns)]);
    typedef void (*Outer)    (const ConvolutionInput <Dim, Chns, Backprop>&, Tensor <float, Dim + ((1 + Backprop) * Chns)>&, uint [Dim + (2 * Chns)]);
    typedef void (*Channel)  (const ConvolutionInput <Dim, Chns, Backprop>&, Tensor <float, Dim + ((1 + Backprop) * Chns)>&, uint [2]);
};

template <size_t Dim, bool Chns, bool Backprop>
void __inner_convolution_loop (const ConvolutionInput <Dim, Chns, Backprop>& conv_inpt, Tensor <float, Dim + ((1 + Backprop) * Chns)>& output, uint index [2 * (Dim + Chns)]) 
{
    // index has form: [(ChOut, ChIn,) Out..., Kernel...]

    size_t dimensions [Dim];
    for (uint i = 0; i < Dim; i++) 
    {
        dimensions [i] = conv_inpt.input.dimensions [i + Chns];
    };

    uint InputIndex [Dim + Chns]; // (Channel), A, B, C...
    uint KernelIndex [Dim + ((2 - Backprop) * Chns)]; // (Channel Out, Channel In), A, B, C...
    uint OutputIndex [Dim + ((1 + Backprop) * Chns)]; // (Channel), A, B, C...

    if (Chns)
    {
        InputIndex [0] = index [1];

        if (!Backprop)
        {
            KernelIndex [0] = index [0];
            KernelIndex [1] = index [1];

            OutputIndex [0] = index [0];
        } 
        else 
        {
            KernelIndex [0] = index [0];

            OutputIndex [0] = index [0];
            OutputIndex [1] = index [1];
        };
    };

    for (uint i = 0; i < Dim; i++)
    {
        //      (output coords   *  downsample factor) +   kernel coords   -     padding
        int j = index [i + (2 * Chns)] * conv_inpt.downsample + index [Dim + (2 * Chns) + i] - conv_inpt.padding [i];

        // Range check: add zero if index out of range to emmulate zero padding
        if (j < 0 || j >= dimensions [i]) return;

        InputIndex [i + Chns] = j;
    };

    for (uint i = 0; i < Dim; i++)
    {
        KernelIndex [i + (2 - Backprop) * Chns] = index [Dim + (2 * Chns) + i];
        OutputIndex [i + ((1 + Backprop) * Chns)] = index [i + (2 * Chns)];
    };

    output [OutputIndex] += conv_inpt.input [InputIndex] * conv_inpt.kernel [KernelIndex];
};

template <size_t Dim, bool Chns, bool Backprop>
void __outer_convolution_loop (const ConvolutionInput <Dim, Chns, Backprop>& conv_inpt, Tensor <float, Dim + ((1 + Backprop) * Chns)>& output, uint index [Dim + (2 * Chns)]) 
{
    size_t kernel_dimensions [Dim];
    for (uint i = 0; i < Dim; i++)
    {
        kernel_dimensions [i] = conv_inpt.kernel.dimensions [i + ((2 - Backprop) * Chns)];
    };

    Iterate 
    <
        typename ConvolutionIterFunc <Dim, Chns, Backprop>::Inner, 
        const ConvolutionInput <Dim, Chns, Backprop>&, 
        Tensor <float, Dim + ((1 + Backprop) * Chns)>&, 
        Dim, 
        Dim + (2 * Chns)
    > 
    (
        __inner_convolution_loop, conv_inpt, output, kernel_dimensions, index
    );
};

template <size_t Dim, bool Chns, bool Backprop>
void __channel_convolution_loop (const ConvolutionInput <Dim, Chns, Backprop>& conv_inpt, Tensor <float, Dim + ((1 + Backprop) * Chns)>& output, uint channels [2])
{
    size_t output_dimensions [Dim];
    for (uint i = 0; i < Dim; i++)
    {
        output_dimensions [i] = output.dimensions [i + ((1 + Backprop) * Chns)];
    };

    Iterate 
    <
        typename ConvolutionIterFunc <Dim, Chns, Backprop>::Outer, 
        const ConvolutionInput <Dim, Chns, Backprop>&, 
        Tensor <float, Dim + ((1 + Backprop) * Chns)>&, 
        Dim, 
        2
    >
    (
        __outer_convolution_loop, conv_inpt, output, output_dimensions, channels
    );
};

// Convolution with channels
template <size_t Dim, bool Chns, bool Backprop>
void Convolve (
    const Tensor <float, Dim + Chns>& input, 
    const Tensor <float, Dim + ((2 - Backprop) * Chns)>& kernel,
          Tensor <float, Dim + ((1 + Backprop) * Chns)>& output, 
    ConvolutionType type = same,
    uint downsample = 1
)
{
    ConvolutionInput <Dim, Chns, Backprop> conv_inpt (input, kernel, type, downsample);

    output.SetElements (0.0);

    size_t channels [2];

    if (Chns)
    {
        if (!Backprop)
        {
            for (uint i = 0; i < 2; i++)
            {
                channels [i] = kernel.dimensions [i];
            };
        } 
        else 
        {
            for (uint i = 0; i < 2; i++)
            {
                channels [i] = output.dimensions [i];
            };
        };

        Iterate 
        <
            typename ConvolutionIterFunc <Dim, Chns, Backprop>::Channel, 
            const ConvolutionInput <Dim, Chns, Backprop>&, 
            Tensor <float, Dim + ((1 + Backprop) * Chns)>&, 
            2
        > 
        (
            __channel_convolution_loop, conv_inpt, output, channels
        );
    }
    else
    {
        Iterate 
        <
            typename ConvolutionIterFunc <Dim, Chns, Backprop>::Outer, 
            const ConvolutionInput <Dim, Chns, Backprop>&, 
            Tensor <float, Dim + ((1 + Backprop) * Chns)>&, 
            Dim
        >
        (
            __outer_convolution_loop, conv_inpt, output, output.dimensions
        );
    };
};

// ***---------  CONVOLUTION LOGIC  ---------***
// if channels, one dimension of the input and the output denotes channel
// the kernel then has one extra dimension to represent whether or not the channels match.
// ie CONVOLVE (DIM, DIM + 1) = DIM
//               1      2        1
//              [        ]
//                     [          ]

// if no channels, no consideration of channels is needed
// ie CONVOLVE (DIM, DIM) = DIM
//     (iterate over output dimensions, then inner)

// during backpropagation, you convolve the input and output to find values in the kernel
// ie CONVOLVE (DIM, DIM) = DIM + 1
//               1    1        2
//              [               ]
//                   [          ]

//     (iterate over both channels, then dimensions)
// channels then dim:    CHout CHin ((Aout Bout) (Ain Bin))


int Kronecker (int i, int j)
{
    return (i == j);
};

float* Identity (float x [], size_t n) 
{
    return x;
};

float Identity (float x)
{
    return x;
};

float ReLU (float x) 
{
    return (x > 0) ? x : 0;
};

float Heaviside (float x, float a) 
{
    return (x + a > 0) ? 1 : 0;
};

float Step (float x) 
{
    return (x > 0) ? 1 : 0;
};

float Sigmoid (float x) 
{
    return 1 / (1 + exp (-x));

};

float SigmoidDerivative (float x)
{
    return Sigmoid (x) * (1 - Sigmoid (x));
};

float Max (float x [], size_t n)
{
    float max = x [0];

    for (int i = 1; i < n; i++)
    {
        if (x [i] > max)
        {
            max = x [i];
        };
    };

    return max;
};

float* Softmax (float x [], size_t n) 
{
    float total = 0.0;
    float stability = - Max (x, n);

    for (int i = 0; i < n; i++)
    {
        total += exp (x [i] + stability);
    };

    float* g = new float [n];

    for (int i = 0; i < n; i++)
    {
        g [i] = exp (x [i] + stability) / total;
    };

    return g;
};

float Softplus (float x);

float GELU (float x);

float ELU (float x);

float Gaussian (float x);


float WeightedSum (float values[], float weights [], float bias, size_t length) 
{
    float accumulated = bias;

    for (int i = 0; i < length; i++) 
    {
        accumulated += values [i] * weights [i];
    };

    return accumulated;
};

// TODO: Fix Gradient functions memory leak (g is not deleted in function)
float MeanSquaredError (float output [], float expected [], size_t n)
{
    float total = 0.0;

    for (int i = 0; i < n; i++)
    {
        float difference = expected [i] - output [i];
        total += pow (difference, 2);
    };


    return total / n;
};

float* MeanSquaredErrorGradient (float output [], float expected [], size_t n) 
{
    float* g = new float [n];

    for (int i = 0; i < n; i++)
    {
        g [i] = - 2 * (expected [i] - output [i]) / n;
    };

    return g;
};

float CrossEntropy (float output [], float expected [], size_t n)
{
    float total = 0.0;
    float epsilon = 0.01;

    for (int i = 0; i < n; i++)
    {
        total += expected [i] * log (output [i] + epsilon);
    };


    return -total;
};

float* CrossEntropyGradient (float output [], float expected [], size_t n) 
{
    float* g = new float [n];

    for (int i = 0; i < n; i++)
    {
        g [i] = output [i] - expected [i];
    };

    return g;
};

float CrossEntropy (const Tensor <float, 3>& output, const Tensor <float, 3>& expected)
{
    float total = 0.0;
    float epsilon = 0.01;

    uint chs = output.dimensions [0];
    uint rows = output.dimensions [1];
    uint cols = output.dimensions [2];

    for (uint i = 0; i < chs; i++)
        for (uint j = 0; j < rows; j++)
            for (uint k = 0; k < cols; k++)
                total += expected [i][j][k] * log (output [i][j][k] + epsilon);

    return -total;
};

template <size_t Dim>
void CrossEntropyGradient (const Tensor <float, Dim>& output, const Tensor <float, Dim>& expected, Tensor <float, Dim>& g) 
{
    for (uint i = 0; i < output.length; i++)
    {
        g.elements [i] = output.elements [i] - expected.elements [i];
    };
};

template <size_t depth>
struct Random 
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution [depth];

    Random (size_t dim []) : generator (rd ()) 
    {
        for (int i = 0; i < depth; i++)
        {
            distribution [i] = std::normal_distribution <float> (0.0, ((float)1) / (float)(dim [i])); // Mean 0, STDEV 1/n
        };
    };
    Random (size_t dim [], int seed) : generator (seed) 
    {
        for (int i = 0; i < depth; i++)
        {
            distribution [i] = std::normal_distribution <float> (0.0, ((float)1) / (float)(dim [i])); // Mean 0, STDEV 1/n
        };
    };

    float RandomWeight (int i) 
    {
        return distribution [i] (generator);
    };
};

template <>
struct Random <1>
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution;

    Random (size_t length) : generator (rd ()) 
    {
        distribution = std::normal_distribution <float> (0.0, ((float)1) / (float)(length)); // Mean 0, STDEV 1/n
    };
    Random (size_t length, int seed) : generator (seed) 
    {
        distribution = std::normal_distribution <float> (0.0, ((float)1) / (float)(length)); // Mean 0, STDEV 1/n
    };

    float RandomWeight () 
    {
        return distribution (generator);
    };
};

template <size_t Dim, bool Chns>
void InitialiseKernel (Random <1>* r, Tensor <float, Dim + (2 * Chns)>* kernel, uint index [Dim + (2 * Chns)])
{
    if (Chns && (index [0] != index [1]))
    {
        (*kernel) [index] = 0.0;
    }
    else
    {
        (*kernel) [index] = r -> RandomWeight ();
        // (*kernel) [index] = 1.0;
    };
};

template <size_t Dim, bool Chns>
struct ConvolutionLayer 
{
    // TODO: make everything general using iterate and convolve

    Tensor <float, Dim + Chns>* output;
    Tensor <float, Dim + (2 * Chns)>* kernel;

    ConvolutionType type;
    uint downsample;

    ConvolutionLayer 
    (
        Tensor <float, Dim + (2 * Chns)>* initial_kernel, 
        size_t input_dim [Dim + Chns], 
        size_t output_dim [Dim + Chns],
        size_t kernel_dim [Dim + (2 * Chns)], 
        Random <1>* r, 
        ConvolutionType type,
        uint downsample
    )
        : type {type}, downsample {downsample}
    {
        output = new Tensor <float, Dim + Chns> (output_dim);
        kernel = new Tensor <float, Dim + (2 * Chns)> (kernel_dim);

        if (initial_kernel == nullptr)
        {
            typedef void (*KernelInitialiser) (Random <1>*, Tensor <float, Dim + (2 * Chns)>*, uint [Dim + (2 * Chns)]);

            Iterate <KernelInitialiser, Random <1>*, Tensor <float, Dim + (2 * Chns)>*, Dim + (2 * Chns)> (InitialiseKernel <Dim, Chns>, r, kernel, kernel_dim);
        }
        else 
        {
            kernel -> SetElements (initial_kernel);
        };
    };

    void Propagate (const Tensor <float, Dim + Chns>& input) 
    {
        Convolve <Dim, Chns, false> (input, (*kernel), (*output), type, downsample);
    };

    //TODO: 
    void BackPropagate (const Tensor <float, Dim + Chns>& input, const Tensor <float, Dim + Chns>& expected) 
    {
        Propagate (input);

        Tensor <float, Dim + Chns> input_gradient (input.dimensions);
        Tensor <float, Dim + Chns> output_gradient (output -> dimensions);
        Tensor <float, Dim + (2 * Chns)> kernel_gradient (kernel -> dimensions);

        Tensor <float, Dim + Chns> flipped_input = input.Copy ();
        Tensor <float, Dim + (2 * Chns)> flipped_kernel = (*kernel).Copy ();

        flipped_input.Flip ();
        flipped_kernel.Flip ();

        CrossEntropyGradient ((*output), expected, output_gradient);

        Convolve <Dim, Chns, false> (output_gradient, flipped_kernel, input_gradient, type, downsample);
        Convolve <Dim, Chns, true>  (output_gradient, flipped_input, kernel_gradient, type, downsample);
        //                              input             kernel         output

        for (uint i = 0; i < kernel -> length; i++)
        {        
            kernel -> elements [i] += 0.01 * kernel_gradient.elements [i];
        };
    };

    #if DEBUG_LEVEL == 1

    void PrintKernel () 
    {
        std::cout << "Kernel: " << std::endl;
        kernel -> Print ();
    };

    void PrintInput (const Tensor <float, Dim + Chns>& input) 
    {
        std::cout << "Input: " << std::endl;
        input.Print ();
    };

    void PrintOutput () 
    {
        std::cout << "Output: " << std::endl;
        output -> Print ();
    };

    #endif
};


template <size_t depth>
struct Layer
{
    float** weights;
    float* biases;
    struct { size_t M, N; } size;

    float* activations;
    float* x;

    activation_fn fn;
    activation_fn fn_prime;
    

    // Constructor
    Layer 
    (
        float** p, float* b, 
        size_t M, size_t N, 
        activation_fn f, activation_fn f_prime, 
        Random <depth>* r, int layer_depth
    ) 
        : fn (f), fn_prime (f_prime)
    {
        size.M = M;
        size.N = N;
        // TODO: make tensor?
        activations = new float [M]();
        x = new float [M]();

        weights = new float* [M];
        weights [0] = new float [M * N];
        for (int i = 1; i < M; i++)
        {
            weights [i] = weights [i - 1] + N; 
        };

        if (p == nullptr) 
        {
            for (int i = 0; i < M * N; i++)
            {
                weights [0][i] = r -> RandomWeight (layer_depth);
            };
        }
        else 
        {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    weights [0][i * N + j] = p [i][j];
        };

        if (b == nullptr)
        {
            biases = new float [M]();
        }
        else
        {
            biases = b;
        };
    };

    Layer () {};

    #if DEBUG_LEVEL == 1
    Layer (const Layer &l) {
        std::cout << "Copying Layer! " << std::endl;
    };
    #endif

    ~Layer ()
    {
        delete [] weights [0];
        delete [] weights;
        delete [] x;
        delete [] biases; //? is this a problem? maybe
        delete [] activations;
    };

    void SetActivations (float input [])
    {
        size_t M = size.M;
        size_t N = size.N;

        for (int i = 0; i < M; i++) 
        {
            x [i] = WeightedSum (input, weights [i], biases [i], N);
            activations [i] = fn (x [i]);
        };
    };
};

template <size_t depth>
struct Network 
{
    Layer <depth>* layers [depth];
    size_t* dimensions;
    float* output;

    output_fn OutputFunction;
    loss_fn LossFunction;
    loss_gradient LossGradient;

    const float regularisation_factor;
    float learning_rate;
    const float base_learning_rate;
    const float learning_rate_time_constant;
    float momentum;
    const float decay_rate;
    const int epochs;

    Random <depth>* r;
    const int seed;

    // TODO: make tensor?
    float** weight_gradients [depth];
    float* bias_gradients [depth];

    float** weight_velocities [depth];
    float* bias_velocities [depth];

    float** weight_RMSP [depth];
    float* bias_RMSP [depth];


    // Constructor
    Network 
    (
        size_t dimensions [depth + 1], 

        activation_fn functions [depth], 
        activation_fn derivatives [depth], 

        output_fn OutputFunction = Identity, 

        loss_fn LossFunction = CrossEntropy, 
        loss_gradient LossGradient = CrossEntropyGradient,
        float regularisation_factor = 0.0,
        float learning_rate = 0.1,
        float learning_rate_time_constant = 200,
        float momentum = 0.9,
        float rms_decay_rate = 0.1,
        int epochs = 1,
        int seed = 1000
    ) 

        // Initialisation List
        :
        dimensions (dimensions), 
        OutputFunction (OutputFunction), 
        LossFunction (LossFunction), 
        LossGradient (LossGradient), 
        regularisation_factor (regularisation_factor),
        learning_rate (learning_rate),
        base_learning_rate (learning_rate),
        learning_rate_time_constant (learning_rate_time_constant),
        momentum (momentum),
        decay_rate (rms_decay_rate),
        epochs (epochs),
        seed (seed)

    // Constructor Body
    {
        output = new float [dimensions [depth]];
        r = new Random <depth> (dimensions, 1000);

        for (int i = 0; i < depth; i++) 
        {
            size_t M = dimensions [i + 1];
            size_t N = dimensions [i];

            activation_fn f = functions [i];
            activation_fn f_prime = derivatives [i];

            // Initialise gradients //TODO: make tensor?
            float** w = new float* [M];
            for (int j = 0; j < M; j++)
            {
                w [j] = new float [N];
            };

            float* b = new float [M];

            weight_gradients [i] = w;
            bias_gradients [i] = b;

            // Initialise velocities
            float** wv = new float* [M];
            for (int j = 0; j < M; j++)
            {
                wv [j] = new float [N]();
            };

            float* bv = new float [M]();

            weight_velocities [i] = wv;
            bias_velocities [i] = bv;

            // Initialise RMSProp accumulation variables
            float** wrmsp = new float* [M];
            for (int j = 0; j < M; j++)
            {
                wrmsp [j] = new float [N]();
            };

            float* brmsp = new float [M]();

            weight_RMSP [i] = wrmsp;
            bias_RMSP [i] = brmsp;

            // Create layer
            layers [i] = new Layer <depth> (nullptr, nullptr, M, N, f, f_prime, r, i);
        };
    };

    ~Network ()
    {
        delete [] output;
        delete r;

        for (int i = 0; i < depth; i++)
        {
            size_t M = dimensions [i + 1];

            for (int j = 0; j < M; j++)
            {
                delete [] weight_gradients [i][j];
                delete [] weight_velocities [i][j];
                delete [] weight_RMSP [i][j];
            };

            delete [] weight_gradients [i];
            delete [] bias_gradients [i];

            delete [] weight_velocities [i];
            delete [] bias_velocities [i];

            delete [] weight_RMSP [i];
            delete [] bias_RMSP [i];

            delete layers [i];
        };
    };

    float* Propagate (float input []) 
    {
        for (int i = 0; i < depth; i++) 
        {
            Layer <depth>* l = layers [i];
            l -> SetActivations (input);

            input = l -> activations;
        };

        Layer <depth>* l = layers [depth - 1];
        size_t M = l -> size.M;
        float* x = OutputFunction (l -> activations, M);

        for (int i = 0; i < M; i++)
        {
            output [i] = x [i];
        };

        return output;
    };

    #if DEBUG_LEVEL == 1
    void PrintLayer (Layer <depth>* l) 
    {
        if (l == nullptr)
        {
            l = layers [depth - 1];
        }

        size_t M = l -> size.M;

        for (int i = 0; i < M; i++)
        {
            std::cout << l -> activations [i] << "  ";
        };

        std::cout << std::endl;
    };

    void PrintOutput ()
    {
        std::cout << "Output: ";

        for (int i = 0; i < dimensions [depth]; i++)
        {
            std::cout << output [i] << " ";
        };
        std::cout << std::endl;
    };

    void PrintAllLayers () 
    {
        for (int i = 0; i < depth; i++)
        {
            std::cout << std::endl << i << std::endl;
            PrintLayer (layers [i]);
        };
    };

    void PrintWeights ()
    {
        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];

            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            std::cout << std::endl << "layer: " << i << std::endl;
            std::cout << "weights: " << std::endl;

            for (int j = 0; j < M; j++)
            {
                std::cout << "    ";
                for (int k = 0; k < N; k++)
                {
                    std::cout << layer -> weights [j][k] << " ";
                };
                std::cout << std::endl;
            };

            std::cout << "biases: ";

            for (int j = 0; j < M; j++)
            {
                std::cout << layer -> biases [j]<< " ";
            };
            std::cout << std::endl;
        };
        std::cout << std::endl << std::endl;
    };

    void test (float* input_set [], float* expected_set [], size_t set_size) 
    {
        size_t input_dimension = dimensions [0];
        size_t output_dimension = dimensions [depth];

        for (int i = 0; i < set_size; i++)
        {
            std::cout << std::endl << "input: ";
            for (int j = 0; j < input_dimension; j++) 
            {
                std::cout << input_set [i][j] << " ";
            };
            std::cout << std::endl;

            PrintOutput ();

            std::cout << "expected: ";
            for (int j = 0; j < output_dimension; j++) 
            {
                std::cout << expected_set [i][j] << " ";
            };
            std::cout << std::endl;
        };
    };
    #endif

    float Regulariser () 
    {
        // float bias_sum = 0.0;
        float weight_sum = 0.0;

        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            float** w = layer -> weights;
            // float* b = layer -> biases;
            

            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            for (int j = 0; j < M; j++)
            {
                // bias_sum += pow (b [j], 2);

                for (int k = 0; k < N; k++)
                {
                    weight_sum += pow (w [j][k], 2);
                };
            };
        };

        // return weight_sum + bias_sum;
        return weight_sum;
    };

    void SetRegulariserGradients (Layer <depth>* layer, float* b, float** w) 
    {
        size_t M = layer -> size.M;
        size_t N = layer -> size.N;

        // float* biases = layer -> biases;
        float** weights = layer -> weights;

        for (int i = 0; i < M; i++)
        {
            // b [i] = 2 * biases [i];
            b [i] = 0;

            for (int j = 0; j < N; j++)
            {
                w [i][j] = 2 * weights [i][j];
            };
        };
    };

    float Cost (float input [], float expected []) 
    {
        float* output = Propagate (input);
        size_t n = dimensions [depth];
        float loss = LossFunction (output, expected, n);

        return loss + regularisation_factor * Regulariser ();
    };

    void UpdateLearningRate (int i)
    {
        float alpha = (float)i / (float)learning_rate_time_constant;
        alpha = std::min (alpha, (float)1.0);
        learning_rate = (1 - 0.99 * alpha) * base_learning_rate;
    };

    float* GD_Basic (float* input_set [], float* expected_set [], size_t set_size, int seed = 1000)
    { 
        float* costs = new float [set_size * epochs];

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < set_size; j++)
            {
                UpdateLearningRate (j);

                int a = indices [j];

                costs [i * set_size + j] = Cost (input_set [a], expected_set [a]);

                BackPropagate (input_set [a], expected_set [a]);
                UpdateGradientDescent ();
            };
        };
        return costs;
    };

    float* GD_Stochastic (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];
        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < k; j++)
            {
                UpdateLearningRate (j);

                costs [i * k + j] = Cost (input_set [indices [j * minibatch_size]], expected_set [indices [j * minibatch_size]]);

                ResetGradients ();

                for (int k = 0; k < minibatch_size; k++)
                {
                    BackPropagateStochastic (input_set [indices [j * minibatch_size + k]], expected_set [indices [j * minibatch_size + k]], mean_batch);
                };

                UpdateGradientDescent ();
            };
        };
        return costs;
    };

    float* GD_StochasticMomentum (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];
        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < k; j++)
            {
                UpdateLearningRate (j);

                costs [i * k + j] = Cost (input_set [indices [j * minibatch_size]], expected_set [indices [j * minibatch_size]]);

                ResetGradients ();

                for (int k = 0; k < minibatch_size; k++)
                {
                    BackPropagateStochastic (input_set [indices [j * minibatch_size + k]], expected_set [indices [j * minibatch_size + k]], mean_batch);
                };

                UpdateMomentum ();
            };
        };
        return costs;
    };

    float* GD_StochasticNesterov (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];
        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < k; j++)
            {
                UpdateLearningRate (j);

                costs [i * k + j] = Cost (input_set [indices [j * minibatch_size]], expected_set [indices [j * minibatch_size]]);

                UpdateInterim ();
                ResetGradients ();

                for (int k = 0; k < minibatch_size; k++)
                {
                    BackPropagateStochastic (input_set [indices [j * minibatch_size + k]], expected_set [indices [j * minibatch_size + k]], mean_batch);
                };

                UpdateMomentum ();
            };
        };
        return costs;
    };

    float* GD_RMSProp (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];
        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < k; j++)
            {
                UpdateLearningRate (j);

                costs [i * k + j] = Cost (input_set [indices [j * minibatch_size]], expected_set [indices [j * minibatch_size]]);

                ResetGradients ();

                for (int k = 0; k < minibatch_size; k++)
                {
                    BackPropagateStochastic (input_set [indices [j * minibatch_size + k]], expected_set [indices [j * minibatch_size + k]], mean_batch);
                };

                UpdateRMSProp ();
            };
        };
        return costs;
    };

    float* GD_RMSPropNesterov (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];
        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        int indices [set_size];
        for (int i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        for (int i = 0; i < epochs; i++)
        {
            shuffle (indices, indices + set_size, std::mt19937 (seed));

            for (int j = 0; j < k; j++)
            {
                UpdateLearningRate (j);

                costs [i * k + j] = Cost (input_set [indices [j * minibatch_size]], expected_set [indices [j * minibatch_size]]);

                UpdateInterim ();
                ResetGradients ();

                for (int k = 0; k < minibatch_size; k++)
                {
                    BackPropagateStochastic (input_set [indices [j * minibatch_size + k]], expected_set [indices [j * minibatch_size + k]], mean_batch);
                };

                UpdateNesterovRMSProp ();
            };
        };
        return costs;
    };

    void ResetGradients ()
    {
        for (int i = 0; i < depth; i++)
        {
            size_t M = dimensions [i + 1];
            size_t N = dimensions [i];

            for (int j = 0; j < M; j++)
            {
                bias_gradients [i][j] = 0.0;

                for (int k = 0; k < N; k++)
                {
                    weight_gradients [i][j][k] = 0.0;
                };
            };
        };
    };

    void BackPropagate (float input [], float expected []) 
    {
        float* y = Propagate (input);
        size_t n = dimensions [depth];
        float* g = LossGradient (y, expected, n);

        // Iterate through layers and calculate gradient
        for (int i = depth - 1; i > -1; i--) 
        {
            Layer <depth>* layer = layers [i];

            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            activation_fn fn_prime = layer -> fn_prime;

            // Gradient of loss function with respect to the nets of layer i
            for (int j = 0; j < M; j++) 
            {
                g [j] *= fn_prime (layer -> x [j]);
            };

            // Prepare some memory
            float b [M];
            float w [M][N];

            float rgb [M];
            float** rgw = new float* [M];
            for (int j = 0; j < M; j++)
            {
                rgw [j] = new float [N];
            };

            // Calculate gradients from regulariser
            SetRegulariserGradients (layer, rgb, rgw);

            // Fetch activations of preivous layer
            float* a;
            if (i > 0) 
            {
                a = layers [i - 1] -> activations;
            }
            else 
            {
                a = input;
            };

            // Calculate gradient of loss function with respect to the weights and biases of layer i
            for (int j = 0; j < M; j++)
            {
                b [j] = g [j] + regularisation_factor * rgb [j];

                for (int k = 0; k < N; k++)
                {
                    w [j][k] = g [j] * a [k] + regularisation_factor * rgw [j][k];
                };
            };

            for (int j = 0; j < M; j++)
            {
                delete [] rgw [j];
            };
            delete [] rgw;

            // Calculate gradient of loss function with respect to the activations of the previous layer (i - 1)
            float* x = new float [N]();
            for (int k = 0; k < N; k++)
            {
                for (int j = 0; j < M; j++)
                {
                    x [k] += g [j] * layer -> weights [j][k];
                };
            };

            delete [] g;
            g = x;

            // Store the gradients
            for (int j = 0; j < M; j++)
            {
                bias_gradients [i][j] = b [j];

                for (int k = 0; k < N; k++)
                {
                    weight_gradients [i][j][k] = w [j][k];
                };
            };
        };
    };

    void BackPropagateStochastic (float input [], float expected [], float mean_batch = 0.0) 
    {
        float* y = Propagate (input);
        size_t n = dimensions [depth];
        float* g = LossGradient (y, expected, n);

        // Iterate through layers and calculate gradient
        for (int i = depth - 1; i > -1; i--) 
        {
            Layer <depth>* layer = layers [i];

            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            activation_fn fn_prime = layer -> fn_prime;

            // Gradient of loss function with respect to the nets of layer i
            for (int j = 0; j < M; j++) 
            {
                g [j] *= fn_prime (layer -> x [j]);
            };

            // Prepare some memory
            float b [M];
            float w [M][N];

            float rgb [M];
            float** rgw = new float* [M];
            for (int j = 0; j < M; j++)
            {
                rgw [j] = new float [N];
            };

            // Calculate gradients from regulariser
            SetRegulariserGradients (layer, rgb, rgw);

            // Fetch activations of preivous layer
            float* a;
            if (i > 0) 
            {
                a = layers [i - 1] -> activations;
            }
            else 
            {
                a = input;
            };

            // Calculate gradient of loss function with respect to the weights and biases of layer i
            for (int j = 0; j < M; j++)
            {
                b [j] = g [j] + regularisation_factor * rgb [j];

                for (int k = 0; k < N; k++)
                {
                    w [j][k] = g [j] * a [k] + regularisation_factor * rgw [j][k];
                };
            };

            for (int j = 0; j < M; j++)
            {
                delete [] rgw [j];
            };
            delete [] rgw;

            // Calculate gradient of loss function with respect to the activations of the previous layer (i - 1)
            float* x = new float [N]();
            for (int k = 0; k < N; k++)
            {
                for (int j = 0; j < M; j++)
                {
                    x [k] += g [j] * layer -> weights [j][k];
                };
            };

            delete [] g;
            g = x;

            // Store the gradients
            for (int j = 0; j < M; j++)
            {
                bias_gradients [i][j] += mean_batch * b [j];

                for (int k = 0; k < N; k++)
                {
                    weight_gradients [i][j][k] += mean_batch * w [j][k];
                };
            };
        };
    };


    void UpdateGradientDescent () 
    { 
        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            
            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            // Update parameters
            for (int j = 0; j < M; j++)
            {
                layer -> biases [j] -= learning_rate * bias_gradients [i][j];

                for (int k = 0; k < N; k++)
                {
                    layer -> weights [j][k] -= learning_rate * weight_gradients [i][j][k];
                };
            };
        };
    };   

    void UpdateMomentum () 
    { 
        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            
            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            // Update velocities
            for (int j = 0; j < M; j++)
            {
                bias_velocities [i][j] = momentum * bias_velocities [i][j] - learning_rate * bias_gradients [i][j];

                for (int k = 0; k < N; k++)
                {
                    weight_velocities [i][j][k] = momentum * weight_velocities [i][j][k] - learning_rate * weight_gradients [i][j][k];
                };
            };

            // Update parameters
            for (int j = 0; j < M; j++)
            {
                layer -> biases [j] += bias_velocities [i][j];

                for (int k = 0; k < N; k++)
                {
                    layer -> weights [j][k] += weight_velocities [i][j][k];
                };
            };
        };
    }; 

    void UpdateInterim () 
    { 
        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            
            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            // Update parameters
            for (int j = 0; j < M; j++)
            {
                layer -> biases [j] += momentum * bias_velocities [i][j];

                for (int k = 0; k < N; k++)
                {
                    layer -> weights [j][k] += momentum * weight_velocities [i][j][k];
                };
            };
        };
    };   

    void UpdateRMSProp () 
    {
        float stabiliser = 0.000001;

        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            
            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            // Update RMSP
            for (int j = 0; j < M; j++)
            {
                bias_RMSP [i][j] = decay_rate * bias_RMSP [i][j] + (1 - decay_rate) * pow (bias_gradients [i][j], 2);

                for (int k = 0; k < N; k++)
                {
                    weight_RMSP [i][j][k] = decay_rate * weight_RMSP [i][j][k] + (1 - decay_rate) * pow (weight_gradients [i][j][k], 2);
                };
            };

            // Update parameters
            for (int j = 0; j < M; j++)
            {
                layer -> biases [j] -= learning_rate * bias_gradients [i][j] / sqrt (stabiliser + bias_RMSP [i][j]);

                for (int k = 0; k < N; k++)
                {
                    layer -> weights [j][k] -= learning_rate * weight_gradients [i][j][k] / sqrt (stabiliser + weight_RMSP [i][j][k]);
                };
            };
        };
    }; 

    void UpdateNesterovRMSProp () 
    { 
        for (int i = 0; i < depth; i++)
        {
            Layer <depth>* layer = layers [i];
            
            size_t M = layer -> size.M;
            size_t N = layer -> size.N;

            // Update RMSP and velocities
            for (int j = 0; j < M; j++)
            {
                bias_RMSP [i][j] = decay_rate * bias_RMSP [i][j] + (1 - decay_rate) * pow (bias_gradients [i][j], 2);
                bias_velocities [i][j] = momentum * bias_velocities [i][j] - learning_rate * bias_gradients [i][j] / sqrt (bias_RMSP [i][j]);

                for (int k = 0; k < N; k++)
                {
                    weight_RMSP [i][j][k] = decay_rate * weight_RMSP [i][j][k] + (1 - decay_rate) * pow (weight_gradients [i][j][k], 2);
                    weight_velocities [i][j][k] = momentum * weight_velocities [i][j][k] - learning_rate  * weight_gradients [i][j][k] / sqrt (weight_RMSP [i][j][k]);
                };
            };

            // Update parameters
            for (int j = 0; j < M; j++)
            {
                layer -> biases [j] += bias_velocities [i][j];

                for (int k = 0; k < N; k++)
                {
                    layer -> weights [j][k] += weight_velocities [i][j][k];
                };
            };

        };
    }; 
};


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

#if DEBUG_LEVEL == 1

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

    T.Print ();
};
    
float test (float x, float y, uint index []) 
{
    std::cout << "test: " << x << std::endl;
    return x;
};

template <size_t N>
void test1 (float in [], float out [], uint index []) 
{   
    for (int i = 0; i < N; i++)
        std::cout << index [i];
    std::cout << std::endl;
};

void test2 (Tensor <float, 3>& in, Tensor <float, 3>& out, uint index []) 
{
    out [index] = in [index];
    out.PrintElements ();
};

void test_iterate ()
{
    typedef float (*F) (float, float, uint []);

    typedef void (*F1) (float[], float[], uint []);

    typedef void (*F2) (Tensor <float, 3>&, Tensor <float, 3>&, uint []);

    size_t dimensions [3] = {2, 3, 4};

    float data = 3.1;
    float out = 0.0;

    float data1 [24];
    float out1 [24];
    for (uint i = 0; i < 24; i++)
        data1 [i] = (i % 2 == 1) ? 1.0 : 2.0;

    Tensor <float, 3> data2 (dimensions, data1);
    Tensor <float, 3> out2 (dimensions);

    Iterate <F, float, float, 3> (test, data, out, dimensions);
    Iterate <F1, float [24], float [24], 3> (test1 <3>, data1, out1, dimensions);
    Iterate <F2, Tensor <float, 3>&, Tensor <float, 3>&, 3> (test2, data2, out2, dimensions);
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

    Convolve <2, true, false> (input, kernel, output, same, 1);

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

    Convolve <3, false, false> (no_CH_input, no_CH_kernel, no_CH_output, same, 1);

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

    Convolve <2, false, false> (small_input, small_kernel, small_output, same, 1);

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

    Convolve <5, false, false> (big_input, big_kernel, big_output, same, 1);

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

    Convolve <6, false, false> (bigger_input, bigger_kernel, bigger_output, same, 1);

    bigger_input.name = "bigger_input";
    bigger_kernel.name = "bigger_kernel";
    bigger_output.name = "bigger_output";

    bigger_input.Print ();
    bigger_kernel.Print ();
    bigger_output.Print ();
};

void test_convolution_layer () 
{
    size_t input_dim [3] = {3, 4, 4};
    size_t output_dim [3] = {3, 4, 4};
    size_t kernel_dim [4] = {3, 3, 2, 2};

    std::mt19937 generator (SEED);
    std::uniform_real_distribution <float> distribution (0.0, 1.0);

    size_t length = 1;
    for (uint i = 0; i < 3; i++)
    {
        length *= input_dim [i];
    };

    float input_elements [length];
    for (uint i = 0; i < length; i++)
    {
        // input_elements [i] = distribution (generator);
        input_elements [i] = 1.0;
    };

    Tensor <float, 3> input (input_dim, input_elements);
    Tensor <float, 3> expected (input_dim, input_elements);

    Random <1>* r = new Random <1> (16, SEED);
    uint downsampling = 1;

    ConvolutionLayer <2, true> layer (nullptr, input_dim, output_dim, kernel_dim, r, same, downsampling);

    layer.Propagate (input);

    layer.PrintInput (input);
    layer.PrintKernel ();
    layer.PrintOutput ();

    layer.BackPropagate (input, expected);
    layer.PrintKernel ();
};

#endif

int main () 
{
    // run_net ();
    // test_tensor ();
    // test_iterate ();
    // test_convolve ();
    test_convolution_layer ();
};

// TODO: combine layers and convolutional layers into network object

/*
Rename Network -> FullyConnectedLayers

New Network contains:   
                ConvolutionalLayers
                FullyConnectedLayers
                RecurrentLayers

Same backpropagation algorithms/structure etc for FullyConnectedLayers, pass final gradient from one stage to the next
*/

// TODO: switch to using Tensor for weights etc
// TODO: check large data not being copied unnecessarily eg pass into functions by reference

// TODO: improve python graphing
// TODO: testing, find decent default values
// TODO: why is RMSProp producing such a weird pattern of losses
// TODO: why is the model not overfitting
// TODO: test on actual data

// TODO: read regularisation chapter

// TODO: begin creating data preprocessing program

// TODO: CUDA? or openCL

// TODO: RNN