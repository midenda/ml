// #include <vector>
#include <cmath>
#include <iostream>

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


float weighted_sum (float values[], float weights [], size_t length) 
{
    float accumulated = 0;

    for (int i = 0; i < length; i++) 
    {
        accumulated += values [i] * weights [i];
    };

    return accumulated;
};

struct Layer
{

    float** weights;
    size_t size [2];

    float* activations;
    activation_fn fn;
    
    Layer (float** p, size_t M, size_t N, activation_fn f) : fn (f)
    {
        size [0] = M;
        size [1] = N;
        activations = new float [M]();

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
                weights [0][i] = RandomWeight ();
            };
        }

        else 
        {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++)
                    weights [0][i * N + j] = p [i][j];
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

    float RandomWeight () 
    {
        return 0;
    };
};


template <size_t depth>
struct Network 
{
    Layer* layers [depth];

    size_t* dim;

    Network (size_t dimensions [depth + 1], activation_fn functions [depth]) 
        : dim (dimensions)
    {
        for (int i = 0; i < depth; i++) 
        {
            size_t M = dim [i + 1];
            size_t N = dim [i];

            activation_fn f = functions [i];

            layers [i] = new Layer (nullptr, M, N, f);
        };
    };

    ~Network ()
    {
        for (int i = 0; i < depth; i++)
        {
            delete layers [i];
        };
    };

    void calc_next_layer (float input [], Layer* l, size_t size [2]) 
    {
        size_t M = size [0];
        size_t N = size [1];

        for (int i = 0; i < M; i++) 
        {
            l -> activations [i] = l -> fn (weighted_sum (input, l -> weights [i], N));
        };
    };

    void propagate (float input []) 
    {
        for (int i = 0; i < depth; i++) 
        {
            Layer* l = layers [i];
            calc_next_layer (input, l, l -> size);

            input = l -> activations;
        };
    };


    void PrintOutput () 
    {
        Layer* l = layers [depth - 1];

        size_t M = l -> size [0];

        for (int i = 0; i < M; i++)
        {
            std::cout << l -> activations [i];
        };

        std::cout << std::endl;
    };
};


int main () 
{
    size_t dimensions [] = {4, 5, 5, 4};
    activation_fn functions [] = {reLU, reLU, reLU};

    float input [4] = {1, 1, 1, 1};

    Network <3> network (dimensions, functions);

    network.propagate (input);
    network.PrintOutput ();
};