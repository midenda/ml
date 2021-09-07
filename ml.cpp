#define DEBUG_LEVEL 0

#include <cmath>
#include <iostream>
#include <fstream>
#include <random>
#include <functional>

typedef float (*activation_fn) (float);
typedef float* (*output_fn) (float[], size_t);
typedef float (*loss_fn) (float[], float[], size_t);
typedef float* (*loss_gradient) (float[], float[], size_t);

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

struct Random 
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution [];

    Random (size_t dim [], size_t depth) : generator (rd ()) 
    {
        for (int i = 0; i < depth; i++)
        {
            distribution [i] = std::normal_distribution <float> (0.0, ((float)1) / (float)(dim [i]));
        };
    };
    Random (size_t dim [], size_t depth, int seed) : generator (seed) 
    {
        for (int i = 0; i < depth; i++)
        {
            distribution [i] = std::normal_distribution <float> (0.0, ((float)1) / (float)(dim [i]));
        };
    };

    float RandomWeight (int i) 
    {
        return distribution [i] (generator);
    };
};


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
        Random* r, int layer_depth
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

    Layer (const Layer &l) {
        std::cout << "Copying Layer! " << std::endl;
    };

    ~Layer ()
    {
        delete [] weights [0];
        delete [] weights;
        delete [] x;
        delete [] biases; // is this a problem? maybe
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
    Layer* layers [depth];
    size_t* dimensions;
    float* output;

    output_fn OutputFunction;
    loss_fn LossFunction;
    loss_gradient LossGradient;

    float regularisation_factor;
    float learning_rate;
    const float base_learning_rate;
    const float learning_rate_time_constant;
    float momentum;
    const float decay_rate;

    Random* r;

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
        float rms_decay_rate = 0.1
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
        decay_rate (rms_decay_rate)

    // Constructor Body
    {
        output = new float [dimensions [depth]];
        r = new Random (dimensions, depth, 1000);

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
            layers [i] = new Layer (nullptr, nullptr, M, N, f, f_prime, r, i);
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
            Layer* l = layers [i];
            l -> SetActivations (input);

            input = l -> activations;
        };

        Layer* l = layers [depth - 1];
        size_t M = l -> size.M;
        float* x = OutputFunction (l -> activations, M);

        for (int i = 0; i < M; i++)
        {
            output [i] = x [i];
        };

        return output;
    };

    void PrintLayer (Layer* l) 
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
            Layer* layer = layers [i];

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

    float Regulariser () 
    {
        // float bias_sum = 0.0;
        float weight_sum = 0.0;

        for (int i = 0; i < depth; i++)
        {
            Layer* layer = layers [i];
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

    void SetRegulariserGradients (Layer* layer, float* b, float** w) 
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

    float* GD_Basic (float* input_set [], float* expected_set [], size_t set_size)
    { 
        float* costs = new float [set_size];

        for (int i = 0; i < set_size; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i], expected_set [i]);

            BackPropagate (input_set [i], expected_set [i], false);
            UpdateGradientDescent ();
        };

        return costs;
    };

    float* GD_Stochastic (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];

        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        for (int i = 0; i < k; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i * minibatch_size], expected_set [i * minibatch_size]);

            ResetGradients ();

            for (int j = 0; j < minibatch_size; j++)
            {
                BackPropagateStochastic (input_set [i * minibatch_size + j], expected_set [i * minibatch_size + j], mean_batch);
            };

            UpdateGradientDescent ();
        };
        return costs;
    };

    float* GD_StochasticMomentum (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];

        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        for (int i = 0; i < k; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i * minibatch_size], expected_set [i * minibatch_size]);

            ResetGradients ();

            for (int j = 0; j < minibatch_size; j++)
            {
                BackPropagateStochastic (input_set [i * minibatch_size + j], expected_set [i * minibatch_size + j], mean_batch);
            };

            UpdateMomentum ();
        };
        return costs;
    };

    float* GD_StochasticNesterov (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];

        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        for (int i = 0; i < k; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i * minibatch_size], expected_set [i * minibatch_size]);

            UpdateInterim ();

            ResetGradients ();

            for (int j = 0; j < minibatch_size; j++)
            {
                BackPropagateStochastic (input_set [i * minibatch_size + j], expected_set [i * minibatch_size + j], mean_batch);
            };

            UpdateMomentum ();
        };
        return costs;
    };

    float* GD_RMSProp (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];

        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        for (int i = 0; i < k; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i * minibatch_size], expected_set [i * minibatch_size]);

            ResetGradients ();

            for (int j = 0; j < minibatch_size; j++)
            {
                BackPropagateStochastic (input_set [i * minibatch_size + j], expected_set [i * minibatch_size + j], mean_batch);
            };

            UpdateRMSProp ();
        };
        return costs;
    };

    float* GD_RMSPropNesterov (float* input_set [], float* expected_set [], size_t set_size, size_t minibatch_size)
    { 
        float* costs = new float [set_size];

        int k = set_size / minibatch_size;
        float mean_batch = (float)1 / (float)minibatch_size;

        for (int i = 0; i < k; i++)
        {
            UpdateLearningRate (i);

            costs [i] = Cost (input_set [i * minibatch_size], expected_set [i * minibatch_size]);

            UpdateInterim ();

            ResetGradients ();

            for (int j = 0; j < minibatch_size; j++)
            {
                BackPropagateStochastic (input_set [i * minibatch_size + j], expected_set [i * minibatch_size + j], mean_batch);
            };

            UpdateNesterovRMSProp ();
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

        // Iterate throug layers and calculate gradient
        for (int i = depth - 1; i > -1; i--) 
        {
            Layer* layer = layers [i];

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

            // Calculate gradient of loss function with respect to the activations of the previous layer (i - 1)
            float* x = new float [N];

            for (int k = 0; k < N; k++)
            {
                x [k] = 0.0;

                for (int j = 0; j < M; j++)
                {
                    x [k] += g [j] * layer -> weights [j][k];
                };
            };

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

        // Iterate throug layers and calculate gradient
        for (int i = depth - 1; i > -1; i--) 
        {
            Layer* layer = layers [i];

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

            // Calculate gradient of loss function with respect to the activations of the previous layer (i - 1)
            float* x = new float [N];

            for (int k = 0; k < N; k++)
            {
                x [k] = 0.0;

                for (int j = 0; j < M; j++)
                {
                    x [k] += g [j] * layer -> weights [j][k];
                };
            };

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
            Layer* layer = layers [i];
            
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
            Layer* layer = layers [i];
            
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
            Layer* layer = layers [i];
            
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
            Layer* layer = layers [i];
            
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

    void UpdateNesterovRMSProp (bool stochastic = true, bool interim = false, bool rms = true) 
    { 
        for (int i = 0; i < depth; i++)
        {
            Layer* layer = layers [i];
            
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


int main () 
{
    // Initialise Network
    size_t dimensions [] = {4, 20, 20, 4};
    activation_fn functions [] = {ReLU, ReLU, ReLU};
    activation_fn derivatives [] = {Step, Step, Step};
    float reg_factor = 0.5;
    float learn_rate = 0.1;
    float learn_rate_time_constant = 300;
    float momentum = 0.5;
    float rms_decay_rate = 0.5;

    Network <3> network (
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
        rms_decay_rate
    );


    // Create Fake Test Data
    size_t size = 10000;
    size_t batch_size = 10;
    float dummy [10000][4];
    float* input [size];
    float* expected [size];

    std::mt19937 generator (1000);
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
    // float* costs = network.GD_StochasticNesterov (input, expected, size, batch_size);
    // float* costs = network.GD_RMSProp (input, expected, size, batch_size);
    float* costs = network.GD_RMSPropNesterov (input, expected, size, batch_size);

    // Process Results
    std::ofstream out;
    out.open ("losses.csv");

    // int num_costs = size;
    int num_costs = size / batch_size;

    for (int i = 0; i < num_costs; i++) 
    {
        out << costs [i] << ",";
    };
    out.close ();

    system ("python graph.py");
};

// TODO: improve python graphing

// TODO: Data oriented refactor - remove branching, hidden state etc