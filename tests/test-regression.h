#include "../regression.h"

void test_regression ()
{
    std::mt19937 generator (SEED);
    std::normal_distribution <float> distribution (0.0, 1.0);

    const size_t size = 1000;

    float x [size];
    float y [size];

    for (uint i = 0; i < size; i++)
    {
        x [i] = ((float)i - (float)size/2.0) / (float)100.0;
        y [i] = - 0.5 * pow (x[i], 4) + 4 * pow (x [i], 3) - 5 * pow (x [i], 2) + x [i] + 3 + distribution (generator);
    };

    #define degree 4

    float* coefficients = new float [degree + 1];

    Regression <float, degree, size> (x, y, coefficients);

    // std::cout << "Coefficients: " << std::endl;
    // for (uint i = 0; i < degree + 1; i++)
    // {
    //     std::cout << coefficients [i] << std::endl;
    // };

    float fit_line [size] = {};
    for (uint i = 0; i < size; i++)
    {
        for (uint j = 0; j < degree + 1; j++)
        {
            fit_line [i] += coefficients [j] * pow (x [i], j);
        };
    };

    // Store values
    std::ofstream out;
    out.open ("regression_test.csv");

    for (uint i = 0; i < size; i++) 
    {
        out << x [i] << ",";
    };
    out << "\n";
    for (uint i = 0; i < size; i++) 
    {
        out << y [i] << ",";
    };
    out << "\n";
    for (uint i = 0; i < size; i++)
    {
        out << fit_line [i] << ",";
    };

    out.close ();

    system ("python3 graph.py regression_test.csv --fit");
};