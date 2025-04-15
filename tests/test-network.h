#include "../ml.h"

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

    system ("python3 graph.py losses.csv");
};