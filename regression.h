
template <typename T>
void SwapRows (T** matrix, uint m, uint n)
{
    T* row = matrix [m];
    matrix [m] = matrix [n];
    matrix [n] = row;
};

template <typename T, size_t N>
void GaussianElimination (T** matrix, float* coefficients) 
{   
    uint pivot_row = 0; // pivot row
    uint pivot_column = 0; // pivot column 

    // Reduce to row echelon form
    while ((pivot_row < N) && (pivot_column < N + 1))
    {
        uint index_max = pivot_row;
        T max = abs (matrix [pivot_row][pivot_column]);

        for (uint i = pivot_row + 1; i < N - pivot_row; i++)
        {
            if (abs (matrix [i][pivot_column]) > max)
            {
                max = abs (matrix [i][pivot_column]);
                index_max = i;
            };
        };

        if (matrix [index_max][pivot_column] == 0.0)
        {
            pivot_column += 1;
        }
        else
        {
            if (index_max != pivot_row)
            {
                SwapRows (matrix, pivot_row, index_max);
            };


            for (uint i = pivot_row + 1; i < N; i++)
            {
                T factor = matrix [i][pivot_column] / matrix [pivot_row][pivot_column];

                matrix [i][pivot_column] = 0.0;
                for (uint j = pivot_column + 1; j < N + 1; j++)
                {
                    matrix [i][j] -= matrix [pivot_row][j] * factor;
                };
            };


            T factor = matrix [pivot_row][pivot_column];
            for (uint i = pivot_column; i < N + 1; i++)
            {
                matrix [pivot_row][i] /= factor;
            };

            pivot_row += 1;
            pivot_column += 1;
        };
    };

    // Back substitution
    // The most complicated nested loop ever - apologies in advance
    for (uint i = N - 1; i > 0; i--)
    {
        for (uint j = i; j < N; j++)
        {
            T factor = matrix [i - 1][j];

            for (uint k = i; k < N + 1; k++)
            {
                matrix [i - 1][k] -= matrix [j][k] * factor;
            };
        };
    };

    // Set coefficients
    for (uint i = 0; i < N; i++)
    {
        coefficients [i] = matrix [i][N];
    };
};

template <typename T, size_t order, size_t N>
void Regression (T x [N], T y [N], float* coefficients)
{
    // Initialise memory for matrix
    T** matrix = new T* [order + 1];
    for (uint i = 0; i < order + 1; i++)
    {
        matrix [i] = new T [order + 2]();
    };

    // First row of the matrix
    matrix [0][0] = N;
    for (uint i = 1; i < order + 1; i++)
    {
        for (uint j = 0; j < N; j++)
        {
            matrix [0][i] += pow (x [j], i);
        };
    };
    
    // Remaining rows of the matrix
    for (uint i = 1; i < order + 1; i++)
    {
        for (uint j = 0; j < order; j++) // skip last and second last columns of matrix
        {
            matrix [i][j] = matrix [i - 1][j + 1];
        };

        // Second last column of matrix
        for (uint j = 0; j < N; j++)
        {
            matrix [i][order] += pow (x [j], order + i);
        };
    };

    // Last column of matrix
    for (uint i = 0; i < order + 1; i++)
    {
        for (uint j = 0; j < N; j++)
        {
            matrix [i][order + 1] += pow (x [j], i) * y [j];
        };
    };

    GaussianElimination <T, order + 1> (matrix, coefficients);
};