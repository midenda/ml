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
void Iterate (void (*f) (const InputType, OutputType, uint* const), const InputType x, OutputType y, size_t dimensions [N], Interim interim);

template <typename FunctionType, typename InputType, typename OutputType, size_t M, size_t N>
void Iterate (void (*f) (const InputType, OutputType, uint* const), const InputType x, OutputType y, size_t dimensions [M], uint* const outer_index, Interim interim);


enum ConvolutionType { valid, optimal, same, full };

typedef float (*activation_fn) (float);
typedef float* (*output_fn) (float[], size_t);
typedef float (*loss_fn) (float[], float[], size_t);
typedef void (*loss_gradient) (float[], float[], float*, size_t);

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

template <typename T, size_t Dim, bool Chns>
T CrossEntropy (Tensor <T, Dim + Chns>& output, const Tensor <T, Dim + Chns>& expected)
{
    float total = 0.0;
    float epsilon = 0.01;

    for (uint i = 0; i < output.length; i++)
    {
        total += 0.5 * expected.elements [i] * log (pow (output.elements [i], 2) + epsilon);
    };

    return -total;
};

template <typename T, size_t Dim, bool Chns>
void CrossEntropyGradient (const Tensor <T, Dim + Chns>& output, const Tensor <T, Dim + Chns>& expected, Tensor <T, Dim + Chns>& g) 
{
    for (uint i = 0; i < output.length; i++)
    {
        g.elements [i] = output.elements [i] - expected.elements [i];
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

    RecurrentLayer (size_t dimension, size_t timesteps, float learning_rate = 0.01) 
        : timesteps {timesteps}, dimension {dimension}, learning_rate {learning_rate}
    {
        size_t dimensions [2] = {timesteps, dimension};

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
        NormalisedRandom <depth>* r, int layer_depth
    ) 
        : fn (f), fn_prime (f_prime)
    {
        size.M = M;
        size.N = N;

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

    NormalisedRandom <depth>* r;
    const int seed; // TODO: use global seed

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
        r = new NormalisedRandom <depth> (dimensions, 1000);

        for (int i = 0; i < depth; i++) 
        {
            size_t M = dimensions [i + 1];
            size_t N = dimensions [i];

            activation_fn f = functions [i];
            activation_fn f_prime = derivatives [i];

            // Initialise gradients
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

    float* GD_Basic (float* input_set [], float* expected_set [], size_t set_size, int seed = 1000) //TODO: use global seed
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

            // Fetch activations of previous layer
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
        float* g = new float [n];
        LossGradient (y, expected, g, n);

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

        delete [] g;
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