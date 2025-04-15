#include "../tensor.h"
#include "../ml.h"

void test_recurrent_layer ()
{
    std::mt19937 generator (SEED);
    std::uniform_real_distribution <float> distribution (0.0, 1.0);

    const size_t timesteps = 4;
    const size_t dimension = 4;
    size_t dimensions [2] = {timesteps, dimension};

    RecurrentLayer <float> layer (dimensions [0], dimensions [1]);

    Tensor <float, 2>** input = new Tensor <float, 2>* [EXAMPLES];
    Tensor <float, 2>** expected = new Tensor <float, 2>* [EXAMPLES];

    for (uint i = 0; i < EXAMPLES; i++)
    {
        float input_elements [timesteps * dimension];
        float expected_elements [timesteps * dimension];

        const float value = distribution (generator);

        for (uint j = 0; j < timesteps; j++)
        {
            for (uint k = 0; k < dimension; k++)
            {
                input_elements [j * timesteps + k] = (k == j) ? value : 0.0;
            };
        };

        for (uint j = 0; j < (timesteps * dimension) - 1; j++)
        {
            expected_elements [j] = (input_elements [j] == 0.0) ? 0.0 : 1.0;
        };

        expected_elements [(timesteps * dimension) - 1] = expected_elements [(timesteps * dimension) - 2];

        input [i] = new Tensor <float, 2> (dimensions, input_elements);
        expected [i] = new Tensor <float, 2> (dimensions, expected_elements);
    };

    // layer.Propagate (input);
    // layer.probabilities -> Print ();

    float costs [EXAMPLES];
    float iteration [EXAMPLES];

    for (uint i = 0; i < EXAMPLES; i++)
    {
        costs [i] = layer.BackPropagate (*(input [i]), *(expected [i]));
        iteration [i] = i / 1000.0;
    };
    // layer.probabilities -> Print ();

    for (uint i = 0; i < 10; i++)
    {
        input [i] -> Print ("Input");
        layer.Propagate (*(input [i]));
        layer.probabilities -> Print ("Probabilities");
        expected [i] -> Print ("Expected");
    };

    float* coefficients = new float [5];
    Regression <float, 5, EXAMPLES> (iteration, costs, coefficients);

    float* fit_line = new float [EXAMPLES];
    FitLine <float, 5, EXAMPLES> (iteration, fit_line, coefficients);

    // Process Results
    std::ofstream out;
    out.open ("losses.csv");

    for (uint i = 0; i < EXAMPLES; i++) 
    {
        out << i << ",";
    };
    out << "\n";
    for (uint i = 0; i < EXAMPLES; i++) 
    {
        out << costs [i] << ",";
    };
    out << "\n";
    for (uint i = 0; i < EXAMPLES; i++) 
    {
        out << fit_line [i] << ",";
    };

    out.close ();

    system ("python3 graph.py losses.csv --fit");

    delete [] coefficients;
    delete [] fit_line;
};