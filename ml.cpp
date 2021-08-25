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

template <size_t length>
float weighted_sum (float (&values)[length], float (&weights) [length]) 
{
    float accumulated = 0;

    for (int i = 0; i < length; i++) {
        accumulated += values [i] * weights [i];
    };

    return accumulated;
};

template <size_t M, size_t N>
struct Layer
{
    float weights [M][N];
    float activations [M];
    activation_fn fn;
    
    Layer (float w [M][N], activation_fn f) : weights (), activations (), fn (f)
    {
        // weights = ;
    };
};


template <size_t depth>
struct Network 
{
    void* layers [depth];
    size_t sizes [depth][2];

    size_t* dim;

    Network (size_t dimensions [depth + 1], activation_fn functions [depth]) 
        : dim (dimensions)
    {
        for (int i = 0; i < depth; i++) {

            size_t M = dim [i + 1];
            size_t N = dim [i];

            activation_fn f = functions [i];

            Layer <M, N>* l = new Layer <M, N> (f);

            layers [i] = static_cast <void*> (l);
            sizes [i][0] = M;
            sizes [i][1] = N;
        };
    };

    template <size_t M, size_t N>
    void calc_next_layer (float input [N], Layer<M, N>* l) 
    {
        for (int i = 0; i < M; i++) {
            l -> activations [i] = l -> f (weighted_sum <N> (input, l -> weights [i]));
        };
    };

    void propagate (int input []) 
    {
        void* p = layers;

        for (int i = 0; i < depth; i++) {

            size_t M = dim [i + 1];
            size_t N = dim [i];

            Layer <M, N>* l = static_cast <Layer <M, N>*> (p);

            calc_next_layer <M, N> (input, l);
            input = l -> activations;

            l++;

            p = static_cast <void*> (l);
        };
    };


    void PrintOutput () 
    {
        size_t M = sizes [depth - 1][0];
        size_t N = sizes [depth - 1][1];
        Layer <M, N>* layer = static_cast <Layer <M, N>*> (layers [depth - 1]);

        std::cout << layer -> activations << std::endl;
    };
};


int main () 
{
    size_t dimensions [] = {4, 5, 5, 4};
    activation_fn functions [] = {reLU, reLU, reLU};

    int input [4] = {1, 1, 1, 1};

    Network <3> network (dimensions, functions);
    network.propagate (input);
    network.PrintOutput ();
};