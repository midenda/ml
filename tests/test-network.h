#define SEED 999
#define DEBUG_LEVEL 1

#include "../tensor.h"
#include "../ml.h"

float polynomial (float x) 
{
    return x / 2;
};

template <typename T1, typename T2>
const T2& (BaseLayer <T1, T2>::* reference) (const T1&) = &BaseLayer <T1, T2>::SetActivations;

template <typename T, typename ReturnType, typename... Args>
ReturnType (T::* ref) (Args... args) = &T::SetActivations;

void test_network ()
{
    ActivationFunction sig (Sigmoid, SigmoidDerivative);
    ActivationFunction relu (ReLU, Step);

    constexpr size_t N = 4;
    constexpr size_t InputDimension = 4;
    constexpr size_t OutputDimension = 4;

    size_t dimensions [N + 1] = {InputDimension, 10, 50, 10, OutputDimension};
    ActivationFunction functions [N] = {sig, sig, sig, sig};
    LearningRate rate (0.1, 1000);
    float regularisation = 1e-5; 

    Network <Layers <FeedForwardLayer, N>> network (rate, regularisation, dimensions, functions);

    // Network <Layers <FeedForwardLayer, N>, Layers <FeedForwardLayer, N>> networks;
    float input [3] = {0.5, 0.5, 0.5}; 
    const Tensor <float, 1> input_tensor (3, input);

    // std::cout << std::endl;
    {
        FeedForwardLayer layer (3, 2, ActivationFunction (Identity, Identity));
        (layer.*reference <Tensor <float, 1>, Tensor <float, 1>>) (input_tensor);

        std::mem_fn (&BaseLayer <Tensor <float, 1>, Tensor <float, 1>>::SetActivations);

    };
    //? layers [i].SetActivations (layers [i - 1].activations);

    // std::cout << std::endl;
    {
        Layers <FeedForwardLayer, 5> { 4, 5, ActivationFunction (Identity, Identity) };
    };

    // std::cout << std::endl;
    {
        Network <FeedForwardLayer, FeedForwardLayer, FeedForwardLayer> networks1 
        { 
            rate, regularisation,
            {
                3, 2, ActivationFunction (Identity, Identity)
            }, 
            {
                3, 2, ActivationFunction (Identity, Identity)
            }, 
            {
                3, 2, ActivationFunction (Identity, Identity)
            } 
        };
    };
    
    // std::cout << std::endl;
    {
        Network <FeedForwardLayer, FeedForwardLayer, FeedForwardLayer> networks2 
        { 
            rate, regularisation, (size_t)3, (size_t)2, ActivationFunction (Identity, Identity)
        };
    };
    
    // Create Fake Test Data
    constexpr size_t size = 100;
    // constexpr size_t batch_size = 10;
    constexpr size_t epochs = 100;
    
    Tensor <float, 1> train_input [size];
    Tensor <float, 1> train_expected [size];
    
    std::mt19937 generator (SEED);
    std::uniform_real_distribution <float> distribution (0.0, 1.0);
    
    for (uint i = 0; i < size; i++)
    {   
        train_input [i].Init (InputDimension);
        train_expected [i].Init (OutputDimension);
        
        for (uint j = 0; j < InputDimension; j++)
        {
            train_input [i].elements [j] = distribution (generator);
        };
        
        for (uint j = 0; j < OutputDimension; j++)
        {
            train_expected [i].elements [j] = polynomial (train_input [i].elements [j]);
        };
    };
    
    Tensor <float, 2> costs          = network .GradientDescent <size, epochs> (train_input, train_expected);
    
    // float* costs = network.GD_Basic              (input, expected, size);
    // float* costs = network.GD_Stochastic         (input, expected, size, batch_size);
    // float* costs = network.GD_StochasticMomentum (input, expected, size, batch_size);
    // float* costs = network.GD_StochasticNesterov (input, expected, size, batch_size);
    // float* costs = network.GD_RMSProp            (input, expected, size, batch_size);
    // float* costs = network.GD_RMSPropNesterov    (input, expected, size, batch_size);
    
    // Process Results
    std::ofstream out;
    out.open ("losses.csv");
    
    int num_costs = size * epochs;
    uint start = 1;
    
    
    for (int i = start; i < num_costs; i++) 
    {
        out << i - start << ",";
    };
    out << "\n";
    for (int i = start; i < num_costs; i++) 
    {
        out << costs.elements [i] << ",";
    };
    out.close ();


    // float elements [InputDimension] = {0.1, 0.4, 0.25, 0.9};
    // Tensor <float, 1> test (InputDimension, elements);
    
    // network.Propagate (test);
    // for (uint i = 0; i < N; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     network.layers [i].activations.Print ("Layer i Activations");
    //     // network.layers [i].weights.Print ("Layer i weights");
    // };
    // for (uint i = 0; i < InputDimension; i++)
    // {
    //     std::cout << polynomial (elements [i]) << " ";
    // };
    // std::cout << std::endl;


    // network.Propagate (train_input [size/2]);
    // for (uint i = 0; i < N; i++)
    // {
    //     std::cout << "i = " << i << std::endl;
    //     network.layers [i].activations.Print ("Layer i Activations");
    //     // network.layers [i].weights.Print ("Layer i weights");
    // };
    // for (uint i = 0; i < InputDimension; i++)
    // {
    //     std::cout << polynomial (train_input [size/2].elements [i]) << " ";
    // };
    // std::cout << std::endl;

    
    
    system ("python3 graph.py losses.csv");
};