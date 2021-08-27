#include <cmath>
#include <iostream>
#include <random>
#include <functional>

typedef float (*activation_fn) (float);


float reLU (float x) 
{
    return (x >= 0) ? x : -x;
};

float heaviside (float x, float a) 
{
    return (x + a > 0) ? 1 : 0;
};

float sigmoid (float x) 
{
    return 1 / (1 + exp (-x));

};

float softplus (float x);

float GELU (float x);

float ELU (float x);

float gaussian (float x);


float weighted_sum (float values[], float weights [], float bias, size_t length) 
{
    float accumulated = bias;

    for (int i = 0; i < length; i++) 
    {
        accumulated += values [i] * weights [i];
    };

    return accumulated;
};

struct Random 
{
    std::random_device rd;
    std::mt19937 generator;
    std::normal_distribution <float> distribution;

    Random () : generator (rd ()), distribution (0.0, 1.0) {};
    Random (int seed) : generator (seed), distribution (0.0, 1.0) {};

    float RandomWeight () 
    {
        return distribution (generator);
    };
};


struct Layer
{

    float** weights;
    float* biases;
    size_t size [2]; // TODO: Refactor into struct size {size_t M, N;};

    float* activations;
    activation_fn fn;
    
    Layer (float** p, float* b, size_t M, size_t N, activation_fn f, Random* r) : fn (f)
    {
        size [0] = M;
        size [1] = N;
        activations = new float [M]();
        biases = new float [M];

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
                weights [0][i] = r -> RandomWeight ();
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
            for (int i = 0; i < M; i++)
            {
                biases [i] = r -> RandomWeight ();
            };
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
    };

    void Set_Activations (float input []) // TODO: Pick a function naming convention
    {
        size_t M = size [0];
        size_t N = size [1];

        for (int i = 0; i < M; i++) 
        {
            activations [i] = fn (weighted_sum (input, weights [i], biases [i], N));
        };
    };
};


template <size_t depth>
struct Network 
{
    Layer* layers [depth];
    size_t* dim;

    Random* r = new Random (1000);

    Network (size_t dimensions [depth + 1], activation_fn functions [depth]) 
        : dim (dimensions)
    {
        for (int i = 0; i < depth; i++) 
        {
            size_t M = dim [i + 1];
            size_t N = dim [i];

            activation_fn f = functions [i];

            layers [i] = new Layer (nullptr, nullptr, M, N, f, r);
        };
    };

    ~Network ()
    {
        for (int i = 0; i < depth; i++)
        {
            delete layers [i];
        };
    };

    void propagate (float input []) 
    {
        for (int i = 0; i < depth; i++) 
        {
            Layer* l = layers [i];
            l -> Set_Activations (input);

            input = l -> activations;
        };
    };


    void PrintLayer (Layer* l) 
    {
        if (l == nullptr)
        {
            l = layers [depth - 1];
        }

        size_t M = l -> size [0];

        for (int i = 0; i < M; i++)
        {
            std::cout << l -> activations [i] << "  ";
        };

        std::cout << std::endl;
    };

    void PrintOutput ()
    {
        std::cout << "Output: ";
        PrintLayer (nullptr);
    };

    void PrintAllLayers () 
    {
        for (int i = 0; i < depth; i++)
        {
            std::cout << std::endl << i << std::endl;
            PrintLayer (layers [i]);
        };
    };
};


int main () 
{
    size_t dimensions [] = {4, 5, 5, 4};
    activation_fn functions [] = {reLU, reLU, reLU};

    float input [4] = {1, 1, 1, 1};

    Network <3> network (dimensions, functions);

    network.propagate (input);
    // network.PrintAllLayers ();
    network.PrintOutput ();
};