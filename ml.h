#pragma once

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
#include "./tuple.h"
// #include "./preprocess.h"

// Implementation of std::conditional
// template <bool, typename T, typename F>
// struct conditional
// {
//     typedef T type;
// };

// template <typename T, typename F>
// struct conditional <false, T, F>
// {
//     typedef F type;
// };

#if DEBUG_LEVEL == 1

template <typename T, size_t N>
void DebugPrinter (const T args [N])
{
    for (uint i = 0; i < N; i++)
    {
        std::cout << args [i] << " ";
    };
    std::cout << std::endl; 
};

template <typename ...Args>
void DebugPrinter (Args&&... args)
{
    (std::cout << ... << args) << std::endl; 
};

#endif


template <typename FunctionType, typename InputType, typename OutputType, size_t N>
void Iterate (void (*f) (const InputType, OutputType, uint* const), const InputType x, OutputType y, size_t dimensions [N], Interim interim);

template <typename FunctionType, typename InputType, typename OutputType, size_t M, size_t N>
void Iterate (void (*f) (const InputType, OutputType, uint* const), const InputType x, OutputType y, size_t dimensions [M], uint* const outer_index, Interim interim);


enum ConvolutionType { valid, optimal, same, full };

typedef float (*activation_fn) (float);
typedef float* (*output_fn) (float[], size_t);
typedef float (*loss_fn) (float[], float[], size_t);
typedef void (*loss_gradient) (float[], float[], float*, size_t);

struct ActivationFunction
{
    activation_fn f;
    activation_fn gradient;

    ActivationFunction () {};
    ActivationFunction (activation_fn f, activation_fn df) : f {f}, gradient {df} {};
};

//TODO: Refactor
template <typename T, size_t Dim, bool Chns>
struct LossFunctionInput
{
    const Tensor <T, Dim + Chns>& output; 
    const Tensor <T, Dim + Chns>& expected; 
    float& total; 
    const float epsilon;

    LossFunctionInput 
    (
        const Tensor <T, Dim + Chns>& output,
        const Tensor <T, Dim + Chns>& expected, 
        float& total, 
        float epsilon
    )
        : output {output}, expected {expected}, total {total}, epsilon {epsilon}
    {};
};


template <typename T, size_t Dim, bool Chns, bool Backprop>
struct ConvolutionInput
{
    const Tensor <T, Dim + Chns>& input;
    const Tensor <T, Dim + ((2 - Backprop) * Chns)>& kernel;

    uint downsample;
    uint padding [Dim];

    ConvolutionInput 
    (
        const Tensor <T, Dim + Chns>& input, 
        const Tensor <T, Dim + ((2 - Backprop) * Chns)>& kernel, 
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

template <typename T, size_t Dim, bool Chns, bool Backprop>
void __inner_convolution_loop (const ConvolutionInput <T, Dim, Chns, Backprop>& conv_inpt, Tensor <T, Dim + ((1 + Backprop) * Chns)>& output, uint* const index) 
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

    // Set correct index values depending on Chns & Backprop
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

    // Calculate remaining input index values, simulate padding
    for (uint i = 0; i < Dim; i++)
    {
        //            (output coords   *  downsample factor)  +       kernel coords          -     padding
        int j = index [i + (2 * Chns)] * conv_inpt.downsample + index [Dim + (2 * Chns) + i] - conv_inpt.padding [i];

        // Range check: add zero if index out of range to emmulate zero-padding
        if (j < 0 || j >= dimensions [i]) return;

        InputIndex [i + Chns] = j;
    };

    // Find remaining kernel & output index values
    for (uint i = 0; i < Dim; i++)
    {
        KernelIndex [i + (2 - Backprop) * Chns] = index [Dim + (2 * Chns) + i];
        OutputIndex [i + ((1 + Backprop) * Chns)] = index [i + (2 * Chns)];
    };

    // Update output
    output [OutputIndex] += conv_inpt.input [InputIndex] * conv_inpt.kernel [KernelIndex];
};

template <typename T, size_t Dim, bool Chns, bool Backprop>
void __outer_convolution_loop (const ConvolutionInput <T, Dim, Chns, Backprop>& conv_inpt, Tensor <T, Dim + ((1 + Backprop) * Chns)>& output, uint* const index) 
{
    size_t kernel_dimensions [Dim];
    for (uint i = 0; i < Dim; i++)
    {
        kernel_dimensions [i] = conv_inpt.kernel.dimensions [i + ((2 - Backprop) * Chns)];
    };

    Iterate 
    <
        const ConvolutionInput <T, Dim, Chns, Backprop>&, 
        Tensor <T, Dim + ((1 + Backprop) * Chns)>&, 
        Dim, 
        Dim + (2 * Chns)
    > 
    (
        __inner_convolution_loop, conv_inpt, output, kernel_dimensions, index
    );
};

template <typename T, size_t Dim, bool Chns, bool Backprop>
void __channel_convolution_loop (const ConvolutionInput <T, Dim, Chns, Backprop>& conv_inpt, Tensor <T, Dim + ((1 + Backprop) * Chns)>& output, uint* const channels)
{
    size_t output_dimensions [Dim];
    for (uint i = 0; i < Dim; i++)
    {
        output_dimensions [i] = output.dimensions [i + ((1 + Backprop) * Chns)];
    };

    Iterate 
    <
        const ConvolutionInput <T, Dim, Chns, Backprop>&, 
        Tensor <T, Dim + ((1 + Backprop) * Chns)>&, 
        Dim, 
        2
    >
    (
        __outer_convolution_loop, conv_inpt, output, output_dimensions, channels
    );
};

// Convolution with channels
template <typename T, size_t Dim, bool Chns, bool Backprop>
void Convolve (
    const Tensor <T, Dim + Chns>& input, 
    const Tensor <T, Dim + ((2 - Backprop) * Chns)>& kernel,
          Tensor <T, Dim + ((1 + Backprop) * Chns)>& output, 
    ConvolutionType type = same,
    uint downsample = 1
)
{
    ConvolutionInput <T, Dim, Chns, Backprop> conv_inpt (input, kernel, type, downsample);

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
            const ConvolutionInput <T, Dim, Chns, Backprop>&, 
            Tensor <T, Dim + ((1 + Backprop) * Chns)>&, 
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
            const ConvolutionInput <T, Dim, Chns, Backprop>&, 
            Tensor <T, Dim + ((1 + Backprop) * Chns)>&, 
            Dim
        >
        (
            __outer_convolution_loop, conv_inpt, output, output.dimensions
        );
    };
};

// ***---------  CONVOLUTION LOGIC  ---------*** //
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


// ***---------  ACTIVATION FUNCTIONS  ---------*** //

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

float TanhDerivative (float x)
{
    return 1 - pow (tanh (x), 2);
};

float ArtanhDerivative (float x)
{
    return 1 / (1 - pow (x, 2));
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

template <typename T, size_t N>
float Max (const Tensor <T, N>& x)
{
    float max = x.elements [0];

    for (int i = 1; i < x.length; i++)
    {
        if (x.elements [i] > max)
        {
            max = x [i];
        };
    };

    return max;
};

void Softmax (float x [], float* y, size_t n) 
{
    float total = 0.0;
    float stability = - Max (x, n);

    for (int i = 0; i < n; i++)
    {
        total += exp (x [i] + stability);
    };

    for (int i = 0; i < n; i++)
    {
        y [i] = exp (x [i] + stability) / total;
    };
};

template <typename T, size_t N>
void Softmax (const Tensor <T, N>& x, Tensor <T, N>& y) 
{
    float total = 0.0;
    float stability = - Max <T, N> (x); //TODO: add a comment here for clarity 

    for (int i = 0; i < x.length; i++)
    {
        total += exp (x.elements [i] + stability);
    };

    for (int i = 0; i < x.length; i++)
    {
        y.elements [i] = exp (x.elements [i] + stability) / total;
    };
};

template <typename T, size_t N>
void SoftmaxJacobian (const Tensor <T, N>& input, Tensor <T, 2 * N>& output)
{
    const size_t M = input.length;

    for (uint i = 0; i < M; i++)
    {
        for (uint j = 0; j < M; j++)
        {
            output.elements [i * M + j] = input.elements [i] * ((i == j) - input.elements [j]);
        };
    };
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

void MeanSquaredErrorGradient (float output [], float expected [], float* gradient, size_t n) 
{
    for (int i = 0; i < n; i++)
    {
        gradient [i] = - 2 * (expected [i] - output [i]) / n;
    };
};

template <typename T, size_t N>
float MeanSquaredError (const Tensor <T, N>& output, const Tensor <T, N>& expected)
{
    float total = 0.0;

    for (int i = 0; i < output.length; i++)
    {
        float difference = expected.elements [i] - output.elements [i];
        total += pow (difference, 2);
    };

    return total / (float)output.length;
};

template <typename T, size_t N>
void MeanSquaredErrorGradient (const Tensor <T, N>& output, const Tensor <T, N>& expected, Tensor <T, N>& gradient) 
{
    float length = (float)output.length;
    for (int i = 0; i < output.length; i++)
    {
        gradient.elements [i] = - 2 * (expected.elements [i] - output.elements [i]) / length;
    };
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

void CrossEntropyGradient (float output [], float expected [], float* gradient, size_t n) 
{
    for (int i = 0; i < n; i++)
    {
        gradient [i] = output [i] - expected [i];
    };
};

// template <typename T, size_t Dim, bool Chns>
// T CrossEntropy (Tensor <T, Dim + Chns>& output, const Tensor <T, Dim + Chns>& expected)
// {
//     float total = 0.0;
//     float epsilon = 0.01;

//     for (uint i = 0; i < output.length; i++)
//     {
//         total += 0.5 * expected.elements [i] * log (pow (output.elements [i], 2) + epsilon);
//     };

//     return -total;
// };

// template <typename T, size_t Dim, bool Chns>
// void CrossEntropyGradient (const Tensor <T, Dim + Chns>& output, const Tensor <T, Dim + Chns>& expected, Tensor <T, Dim + Chns>& g) 
// {
//     for (uint i = 0; i < output.length; i++)
//     {
//         g.elements [i] = output.elements [i] - expected.elements [i];
//     };
// };

template <typename T, size_t N>
T CrossEntropy (const Tensor <T, N>& output, const Tensor <T, N>& expected)
{
    float total = 0.0;
    float epsilon = 0.01;

    for (uint i = 0; i < output.length; i++)
    {
        total += 0.5 * expected.elements [i] * log (pow (output.elements [i], 2) + epsilon);
    };

    return -total;
};

template <typename T, size_t N>
void CrossEntropyGradient (const Tensor <T, N>& output, const Tensor <T, N>& expected, Tensor <T, N>& gradient)
{
    for (uint i = 0; i < output.length; i++)
    {
        gradient.elements [i] = output.elements [i] - expected.elements [i];
    };
};

template <typename T, size_t N>
T NegativeLogLikelyhood (const Tensor <T, N>& output, const Tensor <T, N>& expected)
{
    float total = 0.0;
    float epsilon = 0.01;

    for (uint i = 0; i < output.length; i++)
    {
        total += expected.elements [i] * log (output.elements [i] + epsilon);
    };

    return -total;
};

template <typename T, size_t N>
void NegativeLogLikelyhoodGradient (const Tensor <T, N>& output, const Tensor <T, N>& expected, Tensor <T, N>& gradient) 
{
    for (uint i = 0; i < output.length; i++)
    {
        gradient.elements [i] = expected.elements [i] * (output.elements [i] - 1);
    };
};

template <typename T, size_t N>
float Regulariser (const Tensor <T, N>& input)
{
    float total = 0.01;

    for (uint i = 0; i < input.length; i++)
    {
        total += pow (input.elements [i], 2);
    };
    
    return total;
};

template <size_t depth>
struct NormalisedRandom 
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution [depth];

    NormalisedRandom (size_t dim []) : generator (rd ()) 
    {
        for (int i = 0; i < depth; i++)
        {
            distribution [i] = std::normal_distribution <float> (0.0, ((float)1) / (float)(dim [i])); // Mean 0, STDEV 1/n
        };
    };
    NormalisedRandom (size_t dim [], int seed) : generator (seed) 
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
struct NormalisedRandom <1>
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution;

    NormalisedRandom (size_t length) : generator (rd ()) 
    {
        distribution = std::normal_distribution <float> (0.0, ((float)1) / (float)(length)); // Mean 0, STDEV 1/n
    };
    NormalisedRandom (size_t length, int seed) : generator (seed) 
    {
        distribution = std::normal_distribution <float> (0.0, ((float)1) / (float)(length)); // Mean 0, STDEV 1/n
    };

    float RandomWeight () 
    {
        return distribution (generator);
    };
};

template <typename T, size_t Dim, bool Chns>
void InitialiseKernel (NormalisedRandom <1>* r, Tensor <T, Dim + (2 * Chns)>* kernel, uint index [Dim + (2 * Chns)])
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

// ***---------  NETWORK LAYER DEFINITIONS  ---------*** //

//TODO: more advanced scheduler
//TODO: overload +-*/ etc to avoid "learning_rate.rate"
struct LearningRate
{
    const float time_constant;
    const float base_rate;
    float rate;

    LearningRate (float base_rate, float time_constant)
        : time_constant (time_constant), base_rate (base_rate), rate (base_rate)
    {};

    void Update (uint i)
    {
        float alpha = (float)i / time_constant;

        alpha = std::min (alpha, (float)1.0);

        rate = (1.0 - 0.99 * alpha) * base_rate;
    };

    void Update () {}; //TODO: store i internally ?
    void Reset () {};  //TODO: reset to base
};

template <typename Input, typename Output> //TODO: make compatible with different layer types
struct BaseLayer 
{
    using InputType = Input;
    using OutputType = Output;

    //? does the use of virtual functions incur a runtime cost from use of vtables?
    virtual const OutputType& SetActivations   (const InputType&)                      = 0;
    virtual const OutputType& SetGradients     (const InputType&, OutputType&, float)  = 0;
    virtual void              ResetGradients   ()                                      = 0;
    virtual void              UpdateParameters (LearningRate)                          = 0;
    virtual float             Regulariser      (float)                                 = 0;
    //TODO: add implementation details for other gradient descent algorithms (eg UpdateMomentum)
};

template <typename T>
struct RecurrentLayer
{
    Tensor <T, 2>* x;
    Tensor <T, 2>* activations;
    Tensor <T, 2>* outputs;
    Tensor <T, 2>* probabilities;

    Tensor <T, 2>* input_hidden_weights;
    Tensor <T, 2>* hidden_output_weights;
    Tensor <T, 2>* hidden_hidden_weights;

    Tensor <T, 1>* x_biases;
    Tensor <T, 1>* output_biases;

    size_t timesteps;
    size_t dimension;

    float learning_rate;

    //TODO: use init constructor or new tensor api
    RecurrentLayer (size_t dimension, size_t timesteps, float learning_rate = 0.01) 
        : timesteps {timesteps}, dimension {dimension}, learning_rate {learning_rate}
    {
        size_t dimensions [2] = {timesteps, dimension};

        //TODO: better description: what does x represent?
        x             = new Tensor <T, 2> (dimensions);
        activations   = new Tensor <T, 2> (dimensions);
        outputs       = new Tensor <T, 2> (dimensions);
        probabilities = new Tensor <T, 2> (dimensions);

        size_t weight_dimensions [2] = {dimension, dimension};

        input_hidden_weights  = new Tensor <T, 2> (weight_dimensions);
        hidden_output_weights = new Tensor <T, 2> (weight_dimensions);
        hidden_hidden_weights = new Tensor <T, 2> (weight_dimensions);

        x_biases      = new Tensor <T, 1> (dimension);
        output_biases = new Tensor <T, 1> (dimension);

        input_hidden_weights  -> template Randomise <std::normal_distribution <T>> ();
        hidden_output_weights -> template Randomise <std::normal_distribution <T>> ();
        hidden_hidden_weights -> template Randomise <std::normal_distribution <T>> ();

        // x_biases -> SetElements (0.0);
        // output_biases -> SetElements (0.0);
    };

    void Propagate (const Tensor <T, 2>& input) 
    {
        const size_t M = x -> dimensions [1];

        for (uint i = 0; i < timesteps; i++)
        {
            for (uint j = 0; j < M; j++)
            {
                (*x) [i][j] = (*x_biases) [j];

                for (uint k = 0; k < M; k++)
                {
                    if (i > 0)
                    {
                        (*x) [i][j] += (*hidden_hidden_weights) [j][k] * (*activations) [i - 1][k];
                    };
                    (*x) [i][j] += (*input_hidden_weights) [j][k] * input [i][k];
                };

                (*activations) [i][j] = tanh ((*x) [i][j]);

                T weighted_sum_output = 0;

                for (uint k = 0; k < M; k++)
                {
                    weighted_sum_output += (*hidden_output_weights) [j][k] * (*activations) [i][k];
                };

                (*outputs) [i][j] = (*output_biases) [j] + weighted_sum_output;
            };

            Softmax <T, 1> ((*outputs) [i], (*probabilities) [i]);
        };
    };

    float BackPropagate (const Tensor <T, 2>& input, const Tensor <T, 2>& expected)
    {
        Propagate (input);

        const float loss = NegativeLogLikelyhood <T, 2> ((*probabilities), expected);

        size_t weight_dimensions [2] = {dimension, dimension};

        // Declare gradients of intermediary variables
        Tensor <T, 2> outputs_gradient     (outputs -> dimensions);
        Tensor <T, 2> activations_gradient (activations -> dimensions);
        Tensor <T, 2> x_gradient           (x -> dimensions);
        Tensor <T, 2> input_gradient       (input.dimensions);

        // Declare gradients of weight matrices //* Note: these do not vary with time
        Tensor <T, 2> input_hidden_gradient  (weight_dimensions);
        Tensor <T, 2> hidden_output_gradient (weight_dimensions);
        Tensor <T, 2> hidden_hidden_gradient (weight_dimensions);

        // Declare gradients of bias matrices //* Note: these do not vary with time
        Tensor <T, 1> x_biases_gradient (dimension);
        Tensor <T, 1> output_biases_gradient (dimension);

        NegativeLogLikelyhoodGradient <T, 2> ((*probabilities), expected, outputs_gradient);

        // Backpropagation Through Time
        for (int i = timesteps - 1; i > -1; i--)
        {
            for (uint j = 0; j < dimension; j++)
            {
                output_biases_gradient [j] += outputs_gradient [i][j];

                for (uint k = 0; k < dimension; k++)
                {
                    hidden_output_gradient [j][k] += outputs_gradient [i][j] * (*activations) [i][k];
                };
            };

            for (uint j = 0; j < dimension; j++)
            {
                for (uint k = 0; k < dimension; k++)
                {
                    activations_gradient [i][j] += (*hidden_output_weights) [k][j] * outputs_gradient [i][k]; //? Is transpose because of multiplication order?
                };
            };

            if (i < timesteps - 1)
            {
                for (uint j = 0; j < dimension; j++)
                {
                    for (uint k = 0; k < dimension; k++)
                    {
                        activations_gradient [i][j] += (*hidden_hidden_weights) [k][j] * TanhDerivative ((*activations) [i + 1][k]) * activations_gradient [i + 1][k];
                    };
                };
            };

            for (uint j = 0; j < dimension; j++)
            {
                x_gradient [i][j] = TanhDerivative (activations_gradient [i][j]);
                x_biases_gradient [j] += x_gradient [i][j];
            };

            for (uint j = 0; j < dimension; j++)
            {
                for (uint k = 0; k < dimension; k++)
                {
                    input_hidden_gradient [j][k] += x_gradient [i][j] * input [i][k];
                };
            };

            for (uint j = 0; j < dimension; j++)
            {
                for (uint k = 0; k < dimension; k++)
                {
                    input_gradient [i][j] += (*input_hidden_weights) [k][j] * x_gradient [i][k]; //? Is transpose because of multiplication order?
                };
            };

            for (uint j = 0; j < dimension; j++)
            {
                for (uint k = 0; k < dimension; k++)
                {
                    hidden_hidden_gradient [j][k] += TanhDerivative ((*activations) [i][j]) * activations_gradient [i][j] * input [i][k];
                };
            };
        };

        const float regulariser_input_hidden_weights  = Regulariser <T, 2> ((*input_hidden_weights));
        const float regulariser_hidden_output_weights = Regulariser <T, 2> ((*hidden_output_weights));
        const float regulariser_hidden_hidden_weights = Regulariser <T, 2> ((*hidden_hidden_weights));
        
        const float regulariser_x_biases      = Regulariser <T, 1> ((*x_biases));
        const float regulariser_output_biases = Regulariser <T, 1> ((*output_biases));
        
        // Update weights
        for (uint j = 0; j < dimension; j++)
        {
            for (uint k = 0; k < dimension; k++)
            {
                //TODO: using regularisation incorrectly here
                (*input_hidden_weights)  [j][k] -= learning_rate * input_hidden_gradient  [j][k] / regulariser_input_hidden_weights;
                (*hidden_output_weights) [j][k] -= learning_rate * hidden_output_gradient [j][k] / regulariser_hidden_output_weights;
                (*hidden_hidden_weights) [j][k] -= learning_rate * hidden_hidden_gradient [j][k] / regulariser_hidden_hidden_weights;
            };

            (*x_biases)      [j] -= learning_rate * x_biases_gradient      [j] / regulariser_x_biases;
            (*output_biases) [j] -= learning_rate * output_biases_gradient [j] / regulariser_output_biases;
        };

        return loss;
    };
};

template <typename T, size_t Dim, bool Chns>
struct ConvolutionLayer 
{
    Tensor <T, Dim + Chns>* output;
    Tensor <T, Dim + (2 * Chns)>* kernel;

    ConvolutionType type;
    uint downsample;

    const float base_learning_rate;
    float learning_rate;

    const float regularisation_factor;

    ConvolutionLayer 
    (
        Tensor <T, Dim + (2 * Chns)>* initial_kernel, 
        size_t input_dim [Dim + Chns], 
        size_t output_dim [Dim + Chns],
        size_t kernel_dim [Dim + (2 * Chns)], 
        NormalisedRandom <1>* r, 
        ConvolutionType type,
        uint downsample,
        float base_learning_rate = 1.0,
        float regularisation_factor = 0.001
    )
        :   type {type}, 
            downsample {downsample}, 
            base_learning_rate {base_learning_rate}, 
            learning_rate {base_learning_rate}, 
            regularisation_factor {regularisation_factor}
    {
        output = new Tensor <T, Dim + Chns> (output_dim);
        kernel = new Tensor <T, Dim + (2 * Chns)> (kernel_dim);

        if (initial_kernel == nullptr)
        {
            Iterate <NormalisedRandom <1>*, Tensor <T, Dim + (2 * Chns)>*, Dim + (2 * Chns)> (InitialiseKernel <T, Dim, Chns>, r, kernel, kernel_dim);
        }
        else 
        {
            kernel -> SetElements (initial_kernel);
        };
    };

    void Propagate (const Tensor <T, Dim + Chns>& input) 
    {
        Convolve <T, Dim, Chns, false> (input, (*kernel), (*output), type, downsample);
    };

    float BackPropagate (const Tensor <T, Dim + Chns>& input, const Tensor <T, Dim + Chns>& expected) 
    {
        Propagate (input);

        Tensor <T, Dim + Chns> input_gradient (input.dimensions);
        Tensor <T, Dim + Chns> output_gradient (output -> dimensions);
        Tensor <T, Dim + (2 * Chns)> kernel_gradient (kernel -> dimensions);

        // Tensor <T, Dim + Chns> flipped_input = input.Copy ();
        // Tensor <T, Dim + (2 * Chns)> flipped_kernel = (*kernel).Copy ();

        // flipped_input.Rotate ();
        // flipped_kernel.Flip ();

        MeanSquaredErrorGradient <T, Dim + Chns> (*output, expected, output_gradient);
        const float loss = MeanSquaredError <T, Dim + Chns> (*output, expected);
        // float loss = MeanSquaredError <T, Dim + Chns> (*output, expected) + Regulariser <T, Dim + (2 * Chns)> (*kernel);

        // Convolve <T, Dim, Chns, false> (output_gradient, (*kernel), input_gradient, type, downsample);
        // expected.Print ("expected");
        // (*output).Print ("output");

        // output_gradient.Print ("output gradient");

        // TODO: 
        // output_gradient.Rotate ();
        // ? some kind of issue with orientation - seems to be updating the wrong indexes
        // ? do the maths on this
        Convolve <T, Dim, Chns, true>  (output_gradient, input, kernel_gradient, type, downsample);
        //                              input             kernel         output

        // (*kernel).Print ();

        // kernel_gradient.Print ("kernel gradient");
        // kernel_gradient.Rotate ();
        // ? this fixes it for some reason, note swapped order of argumments in Convolve (input, output_gradient)

        for (uint i = 0; i < kernel -> length; i++)
        {     
            // ? should this regularisation be a divide instead of a multiply?
            kernel -> elements [i] -= learning_rate * (kernel_gradient.elements [i]) + regularisation_factor * (kernel -> elements [i]);
        };

        return loss;
    };

    #if DEBUG_LEVEL == 1

    void PrintKernel () 
    {
        kernel -> Print ("Kernel");
    };

    void PrintInput (const Tensor <T, Dim + Chns>& input) 
    {
        input.Print ("Input");
    };

    void PrintOutput () 
    {
        output -> Print ("Output");
    };

    #endif
};

struct FeedForwardLayer : virtual BaseLayer <Tensor <float, 1>, Tensor <float, 1>>
{
    Tensor <float, 2> weights;
    Tensor <float, 1> biases;
    Tensor <float, 1> activations;

    // Gradients are stored in the layer to allow for training algorithms
    // like momentum or RMSProp to introduce a time dependency.
    Tensor <float, 2> weight_gradients;
    Tensor <float, 1> bias_gradients;

    ActivationFunction activation_function;

    FeedForwardLayer () {};

    FeedForwardLayer (const Tensor <float, 2>& w, const Tensor <float, 1>& b, ActivationFunction act_fn)
    {
        Init (w, b, act_fn);
    };

    // TODO: template constructor with new Tensor API (dimension)
    void Init (const Tensor <float, 2>& w, const Tensor <float, 1>& b, ActivationFunction act_fn)
    {
        weights          .Init (w.dimensions, w.elements);
        biases           .Init (b.length,     b.elements);
        weight_gradients .Init (w.dimensions);
        bias_gradients   .Init (w.dimensions [0]);
        activations      .Init (w.dimensions [0]);
        activation_function = act_fn; 
    };

    FeedForwardLayer (size_t M, size_t N, ActivationFunction act_fn)
    {
        // std::cout << "Constructing Layer! " << std::endl;
        Init (M, N, act_fn);
    };
    
    void Init (size_t M, size_t N, ActivationFunction act_fn)
    {
        // std::cout << "Initialising Layer with M=" << M << " and N=" << N << std::endl;

        size_t dimensions [2] = {M, N};

        weights          .Init (dimensions); 
        biases           .Init (M);
        weight_gradients .Init (dimensions);
        bias_gradients   .Init (M);
        activations      .Init (M);

        weights .template Randomise <std::normal_distribution <float>> (0, 1);
        biases  .template Randomise <std::normal_distribution <float>> (0, 1);
        
        activation_function = act_fn;
    };

    #if DEBUG_LEVEL == 1
    FeedForwardLayer (const FeedForwardLayer&) = delete;
    // {
    //     std::cout << "Copying Layer! " << std::endl;
    // };
    #else
        FeedForwardLayer (const FeedForwardLayer&) = delete;
    #endif

    // FeedForwardLayer (FeedForwardLayer&& other) = delete;
    FeedForwardLayer (FeedForwardLayer&& other)
    {
        // std::cout << "Moving Layer! " << std::endl;
        weights          .Init (static_cast <Tensor <float, 2>&&> (other.weights));
        biases           .Init (static_cast <Tensor <float, 1>&&> (other.biases));
        weight_gradients .Init (static_cast <Tensor <float, 2>&&> (other.weight_gradients));
        bias_gradients   .Init (static_cast <Tensor <float, 1>&&> (other.bias_gradients));
        activations      .Init (static_cast <Tensor <float, 1>&&> (other.activations));
        activation_function = other.activation_function; 
    };

    ~FeedForwardLayer () 
    {
        // std::cout << "Deleting Layer! " << std::endl;
    };

    virtual const OutputType& SetActivations (const Tensor <float, 1>& input) override
    {
        for (uint i = 0; i < activations.length; i++)
        {
            float sum = 0.0;
            for (uint j = 0; j < weights.dimensions [1]; j++)
            {
                sum += weights [i][j] * input [j];
            };

            activations [i] = activation_function.f (sum);
        };

        return activations;
    };

    virtual const OutputType& SetGradients (const Tensor <float, 1>& previous_activations, Tensor <float, 1>& gradient, float regularisation_factor) override
    {
        for (uint i = 0; i < gradient.length; i++)
        {
            gradient [i] *= activation_function.gradient (activations [i]);

            bias_gradients [i] = gradient [i] + 2 * regularisation_factor * biases [i]; 
            
            for (uint j = 0; j < weights.dimensions [1]; j++)
            {
                weight_gradients [i][j] = gradient [i] * previous_activations [j] + 2 * regularisation_factor * weights [i][j];
            };

        };

        float new_gradient [previous_activations.length];

        for (uint i = 0; i < previous_activations.length; i++)
        {
            new_gradient [i] = 0.0;
            for (uint j = 0; j < gradient.length; j++)
            {
                new_gradient [i] += gradient [j] * weights [j][i]; 
            };
        };

        // TODO: avoid using Reshape ?
        gradient.Reshape (previous_activations.length, new_gradient);

        return activations;
    };

    virtual void ResetGradients () override {};

    virtual void UpdateParameters (LearningRate learning_rate) override
    {
        float rate = learning_rate.rate; //TODO: improve api
        for (uint i = 0; i < weights.length; i++)
        {
            weights.elements [i] -= rate * weight_gradients.elements [i];
        };

        for (uint i = 0; i < biases.length; i++)
        {
            biases.elements [i] -= rate * bias_gradients.elements [i];
        };
    };

    virtual float Regulariser (float total) override
    {
        for (uint i = 0; i < weights.length; i++)
        {
            total += pow (weights.elements [i], 2);
        };

        return total;
    };
};

template <typename Layer, size_t N>
struct Layers : virtual BaseLayer <Tensor <float, 1>, Tensor <float, 1>> //? requires std::is_base_of <BaseLayer, Layer>::value ?
{
    using LossFunctionType = float (*) (const OutputType&, const OutputType&);
    using LossGradientType = void  (*) (const OutputType&, const OutputType&, OutputType&);

    Layer layers [N];
    LossFunctionType LossFunction;
    LossGradientType LossGradient;

    Layers () {};

    template <typename... Args>
    Layers (Args... args)
        : LossFunction (MeanSquaredError <float, 1>), 
          LossGradient (MeanSquaredErrorGradient <float, 1>)
    {
        for (uint i = 0; i < N; i++)
        {
            layers [i].Init (args...);
        };
    };

    Layers 
    (
        size_t dimensions [N + 1], 
        ActivationFunction functions [N], 
        LossFunctionType f = MeanSquaredError <float, 1>,
        LossGradientType df = MeanSquaredErrorGradient <float, 1>
    )
        : LossFunction (f), LossGradient (df)
    {
        Init (dimensions, functions);
    };

    void Init 
    (
        size_t dimensions [N + 1], 
        ActivationFunction functions [N], 
        LossFunctionType f,
        LossGradientType df
    )
    {
        LossFunction (f);
        LossGradient (df);
        Init (dimensions, functions);
    };

    void Init 
    (
        size_t dimensions [N + 1], 
        ActivationFunction functions [N]
    )
    {
        for (uint i = 0; i < N; i++)
        {
            layers [i].Init (dimensions [i + 1], dimensions [i], functions [i]);
        };
    };

    virtual const OutputType& SetActivations (const InputType& input) 
    override
    {
        layers [0].SetActivations (input);

        for (uint i = 1; i < N; i++)
        {
            layers [i].SetActivations (layers [i - 1].activations);
        };

        return layers [N - 1].activations;
    };

    virtual const OutputType& SetGradients (const InputType& input, Tensor <float, 1>& gradient, float regularisation_factor) 
    override
    {
        for (uint i = N - 1; i > 0; i--)
        {
            layers [i].SetGradients (layers [i - 1].activations, gradient, regularisation_factor);
        };
        layers [0].SetGradients (input, gradient, regularisation_factor);

        return layers [0].activations;
    };
    
    virtual void UpdateParameters (LearningRate rate) 
    override
    {
        for (uint i = 0; i < N; i++)
        {
            layers [i].UpdateParameters (rate);
        }; 
    };

    virtual void ResetGradients () 
    override
    {
        for (uint i = 0; i < N; i++)
        {
            layers [i].ResetGradients ();
        };
    };

    virtual float Regulariser (float total) override
    {
        for (uint i = 0; i < N; i++)
        {
            total += layers [i].Regulariser (total);
        };

        return total;
    };
};


template <typename... Layers> requires (std::is_base_of_v <BaseLayer <Tensor <float, 1>, Tensor <float, 1>>, Layers> && ...)
class Network : private Tuple <Layers...>
{
// Type definitions and Tuple specialisations
private:
    using Base = BaseLayer <Tensor <float, 1>, Tensor <float, 1>>;
    static constexpr const uint N = sizeof... (Layers);

    template <uint I>
    Tuple <Layers...> :: template Type <I>& Get () 
    {
        return Tuple <Layers...> :: template Get <I> ();
    };

// Private data members
private:
    //TODO: refactor learning rate 
    LearningRate learning_rate; // { 0.1, 1000 };
    float regularisation_factor; // = (float)1e-5;

public:
    Network (LearningRate rate, float regularisation, Layers&&... args) 
        : Tuple <Layers...> (static_cast <Layers&&> (args)...), 
            learning_rate { rate }, regularisation_factor { regularisation }
    {
    };

    template <typename... Args>
    Network (LearningRate rate, float regularisation, Args... args) 
        : Tuple <Layers...> (args...),
            learning_rate { rate }, regularisation_factor { regularisation }
    {
    };

    
    Network () = delete; // explicitly delete default constructor

    const typename Base::OutputType& Propagate (const typename Base::InputType& input)
    {
        return Tuple <Layers...>::Propagate (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::SetActivations, input);
    };
    
    const typename Base::OutputType& BackPropagate (const Base::InputType& input, const Base::OutputType& output, const Base::OutputType& expected) 
    {
        Base::OutputType gradient (expected.dimensions);

        //TODO: improve readability of Tuple <Layers...>::template Get <> ()
        Get <N - 1> ().LossGradient (output, expected, gradient);

        return Tuple <Layers...>::BackPropagate (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::SetGradients, input, gradient, regularisation_factor);
        // Tuple <Layers...>::ForEach (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::ResetGradients);
        
    };
    
    void UpdateParameters () 
    {
        Tuple <Layers...>::ForEach (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::UpdateParameters, learning_rate);
    };

    void UpdateHyperParameters () {}; //TODO:

    float Regulariser ()
    {
        return Tuple <Layers...>::Propagate (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::Regulariser, (float)0.01);
    };

    float Cost (const Base::OutputType& output, const Base::OutputType& expected) 
    {
        float loss = Get <N - 1> ().LossFunction (output, expected);
        return loss + regularisation_factor * Regulariser (); //? regulariser per layer or per network?
    };

    //? might have some difficulties using templates if data read from file
    template <size_t set_size, size_t epochs>
    Tensor <float, 2> GradientDescent (const typename Base::InputType input_set [set_size], const typename Base::OutputType expected_set [set_size])
    {
        const size_t dimensions [2] = { epochs, set_size };
        Tensor <float, 2> costs (dimensions);

        uint indices [set_size];
        for (uint i = 0; i < set_size; i++)
        {
            indices [i] = i;
        };

        uint progress = 0;
        
        for (uint i = 0; i < epochs; i++)
        {
            std::shuffle (indices, indices + set_size, std::mt19937 (SEED));
            
            for (uint j = 0; j < set_size; j++)
            {
                learning_rate.Update (j + i * set_size);
                
                uint index = indices [j];
                
                const typename Base::OutputType& output = Propagate (input_set [index]);
                
                costs [i][j] = Cost (output, expected_set [index]);
                
                BackPropagate (input_set [index], output, expected_set [index]);
                UpdateParameters ();
                
                progress = (j + i * set_size) * 100 / (epochs * set_size);
                if (((progress * 10) % 10) == 0)
                {
                    std::cout << '\r' << "progress: ";
                    for (uint k = 0; k < progress/2; k++)
                    {
                        std::cout << '#';
                    };

                    for (uint k = 0; k < 50 - progress/2; k++)
                    {
                        std::cout << ' ';
                    };

                    std::cout << progress << '%';
                    std::cout << std::flush;
                }; 
            }; 
        };
        std::cout << std::flush << "\r\e[K" << "Training Complete!" << std::endl;

        // Returned by copy elision 
        return costs; 
    };

    //TODO: implement other gradient descent algorithms
    Tensor <float, 2> GD_Basic () {};
    Tensor <float, 2> GD_Stochastic () {};
    Tensor <float, 2> GD_StochasticMomentum () {}; 
    Tensor <float, 2> StochasticNesterov () {};
    Tensor <float, 2> GD_RMSProp () {};
    Tensor <float, 2> GD_RMSPropNesterov () {};
};