#include "../benchmark.h"

void print_prime (uint N)
{
    Profile;

    if (N < 1)
    {
        return;
    };

    uint primes [N];

    primes [0] = 2;
    uint found = 1;

    uint i = 3;

    while (found < N)
    {
        for (uint j = 0; j < found; j++)
        {
            if (i % primes [j] == 0) break;

            if (j == found - 1)
            {
                primes [found] = i;
                found++;
            };
        };

        i++;
    };

    std::cout << primes [N - 1] << std::endl;
};

void print_primes (uint N)
{
    Profile;

    for (uint i = 1; i < N + 1; i++)
    {
        print_prime (i);
    };
};

void test_benchmark ()
{
    BenchmarkSession;
    print_primes (100);
};