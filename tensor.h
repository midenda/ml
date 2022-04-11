#pragma once
#include <cmath>

#define DEBUG_LEVEL 1

#if DEBUG_LEVEL == 1
    #include <string>
#endif

#define PrintName(tensor) std::cout << "Printing: " << (#tensor) << "... " << std::endl;

typedef unsigned int uint;

typedef void (*Interim) (uint, size_t);

void nullfn (uint layer, size_t N) {};

void alternating_sort (const uint input [], uint output [], size_t N)
{   
    for (uint i = 0; i < N / 2; i++)
    {
        output [i] = input [2 * i + (N % 2)];
    };
    for (uint i = N / 2; i < N; i++)
    {
        output [i] = input [2 * (i - (N / 2)) + (1 - (N % 2))];
    };
};

void alternating_sort (const size_t input [], size_t output [], size_t N)
{   
    for (uint i = 0; i < N / 2; i++)
    {
        output [i] = input [2 * i + (N % 2)];
    };
    for (uint i = N / 2; i < N; i++)
    {
        output [i] = input [2 * (i - (N / 2)) + (1 - (N % 2))];
    };
};

void undo_alternating_sort (const uint input [], uint output [], size_t N)
{   
    for (uint i = 0; i < N / 2; i++)
    {
        output [2 * i + (N % 2)] = input [i];
    };
    for (uint i = N / 2; i < N; i++)
    {
        output [2 * (i - (N / 2)) + (1 - (N % 2))] = input [i];
    };
};

void undo_alternating_sort (const size_t input [], size_t output [], size_t N)
{   
    for (uint i = 0; i < N / 2; i++)
    {
        output [2 * i + (N % 2)] = input [i];
    };
    for (uint i = N / 2; i < N; i++)
    {
        output [2 * (i - (N / 2)) + (1 - (N % 2))] = input [i];
    };
};

template <typename T, size_t N>
struct Tensor;


template <typename FunctionType, typename InputType, typename OutputType, size_t N>
void __iteration 
(
    FunctionType f, 
    InputType x, 
    OutputType y, 
    size_t dimensions [N], 
    uint* index, 
    uint l, 
    uint offset = 0, 
    Interim interim = nullfn
)
{
    if (l < N)
    {
        for (uint i = 0; i < dimensions [l]; i++)
            __iteration <FunctionType, InputType, OutputType, N> (f, x, y, dimensions, index, l + 1, offset, interim);

        interim (l, N);
    } 
    else 
    {
        f (x, y, index);

        for (uint i = offset + N; i > offset; i--)
        {
            if (index [i - 1] < dimensions [i - offset - 1] - 1)
            {
                index [i - 1] += 1;
                break;
            }
            else
            {
                index [i - 1] = 0;
            };
        };
    };
};

template <typename FunctionType, typename InputType, typename OutputType, size_t N>
void Iterate (FunctionType f, InputType x, OutputType y, size_t dimensions [N], Interim interim = nullfn)
{
    uint l = 0;
    uint* index = new uint [N]{};

    __iteration <FunctionType, InputType, OutputType, N> (f, x, y, dimensions, index, l, 0, interim);
};

template <typename FunctionType, typename InputType, typename OutputType, size_t M, size_t N>
void Iterate (FunctionType f, InputType x, OutputType y, size_t dimensions [M], uint outer_index [N], Interim interim = nullfn)
{
    uint l = 0;
    uint* index = new uint [M + N]{};

    for (uint i = 0; i < N; i++)
    {
        index [i] = outer_index [i];
    };

    __iteration <FunctionType, InputType, OutputType, M> (f, x, y, dimensions, index, l, N, interim);
};

void print_separator (uint layer, size_t N)
{
    if (layer < N)
    {
        if (layer > (N / 2))
        {
            std::cout << "\t";
        }
        else
        {
            std::cout << std::endl;
        };
    };
}; 

template <typename T, size_t N>
struct PrintInput
{
    const Tensor <T, N>& tensor;
    size_t truncation_length;
    size_t spacing;

    PrintInput (const Tensor <T, N>& tensor, size_t truncation_length, size_t spacing)
        : tensor {tensor}, truncation_length {truncation_length}, spacing {spacing}
    {};
};

template <typename T, size_t N>
struct PrintFunction
{
    typedef void (*f) (const PrintInput <T, N>&, void*, uint []);
};

template <typename T, size_t N>
void PrintElement (const PrintInput <T, N>& input, void* output, uint idx [N])
{
    const Tensor <T, N>& tensor = input.tensor;
    size_t truncation_length = input.truncation_length;
    size_t spacing = input.spacing;

    uint index [N];
    undo_alternating_sort (idx, index, N);

    // Crops each element (to 8 characters long) so display is uniform
    std::string character = std::to_string (tensor [index]).substr (0, truncation_length);

    std::cout << character;

    for (uint i = 0; i < truncation_length - character.length () + spacing; i++)
    {
        std::cout << " ";
    };
};

template <typename T, size_t N>
void PrintTensor (const Tensor <T, N>& tensor, size_t truncation_length = 8, size_t spacing = 2)
{
    PrintInput <T, N> input (tensor, truncation_length, spacing);

    size_t dim [N];
    alternating_sort (tensor.dimensions, dim, N);

    Iterate <typename PrintFunction <T, N>::f, PrintInput <T, N>, void*, N> (PrintElement, input, nullptr, dim, print_separator);
};

template <typename T, size_t N>
struct Tensor
{
    size_t dimensions [N];
    T* elements;
    size_t length;
    uint layer;

    Tensor <T, N - 1>** children;

    #if DEBUG_LEVEL == 1

    const char* name;

    #endif

    Tensor () {};

    // Parent Constructor
    Tensor (const size_t input_dimensions [N], T e [])
    {
        layer = 0;

        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        length = dimensions [0];
        size_t dim [N - 1];
        size_t separation = 1;

        for (int i = 0; i < N - 1; i++)
        {
            size_t d = dimensions [i + 1];

            dim [i] = d;
            separation *= d;
            length *= d;
        };

        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };

        size_t l = dimensions [0];
        children = new Tensor <T, N - 1>* [l];

        for (int i = 0; i < l; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (elements + separation * i), 1);
        };
    };

    // Child Constructor: shouldn't be called explicitly
    Tensor (const size_t input_dimensions [N], T* e, uint layer)
        : layer {layer}
    {
        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        size_t child_count = dimensions [0];

        size_t dim [N - 1];
        length = 1;
        for (int i = 0; i < N - 1; i++)
        {
            dim [i] = dimensions [i + 1];
            length *= dimensions [i + 1];
        };

        children = new Tensor <T, N - 1>* [child_count];

        for (int i = 0; i < child_count; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (e + length * i), layer + 1);
        };

        elements = e; 
    };

    Tensor (const size_t input_dimensions [N])
    {
        layer = 0;

        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        length = dimensions [0];
        size_t dim [N - 1];
        size_t separation = 1;

        for (int i = 0; i < N - 1; i++)
        {
            size_t d = dimensions [i + 1];

            dim [i] = d;
            separation *= d;
            length *= d;
        };

        elements = new T [length]{};

        size_t l = dimensions [0];
        children = new Tensor <T, N - 1>* [l];

        for (int i = 0; i < l; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (elements + separation * i), 1);
        };
    };

    ~Tensor () 
    {
        if (children != nullptr)
        {
            for (size_t i = 0; i < dimensions [0]; i++)
            {
                delete children [i];
            };
            delete [] children;
        };

        if  (layer == 0)
        {
            delete [] elements;
        };
    };

    Tensor (const Tensor& t) = delete;

    T& index (uint indices [N])
    {
        uint position = 0;

        for (uint i = 0; i < N; i++)
        {
            uint offset = 1;

            for (uint j = 0; j < N - 1 - i; j++)
            {
                offset *= dimensions [N - 1 - j];
            };

            position += offset * indices [i];
        };
        return elements [position];
    };

    const T& index (uint indices [N]) const
    {
        uint position = 0;

        for (uint i = 0; i < N; i++)
        {
            uint offset = 1;

            for (uint j = 0; j < N - 1 - i; j++)
            {
                offset *= dimensions [N - 1 - j];
            };

            position += offset * indices [i];
        };
        return elements [position];
    };

    Tensor <T, N - 1>& operator[] (uint idx) 
    {
        return *(children [idx]);
    };

    const Tensor <T, N - 1>& operator[] (uint idx) const 
    {
        return *(children [idx]);
    };

    T& operator[] (uint indices [N])
    {
        return index (indices);
    };

    const T& operator[] (uint indices [N]) const 
    {
        return index (indices);
    };

    void SetElements (const Tensor <T, N>& input)
    {
        if (input.length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input.elements [i];
            };
        };
    };

    void SetElements (const Tensor <T, N>* input)
    {
        if (input -> length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input -> elements [i];
            };
        };
    };

    void SetElements (const T input [], size_t input_length)
    {
        if (input_length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input [i];
            };
        };
    };

    void SetElements (T input)
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = input;
        };
    };

    void Flip ()
    {
        T flipped_elements [length];
        for (uint i = 0; i < length; i++)
        {
            flipped_elements [i] = elements [length - i];
        };

        size_t flipped_dimensions [N];
        for (uint i = 0; i < N; i++)
        {
            flipped_dimensions [i] = dimensions [N - i];
        };

        if (children != nullptr)
        {
            for (size_t i = 0; i < dimensions [0]; i++)
            {
                delete children [i];
            };
            delete [] children;
        };

        if  (layer == 0)
        {
            delete [] elements;
        };

        Tensor (flipped_dimensions, flipped_elements);
    };

    const Tensor <T, N> Copy () const
    {
        return Tensor (dimensions, elements);
    };


    #if DEBUG_LEVEL == 1

    void Print () const
    {
        size_t truncation_length = 2;
        size_t spacing = 2;

        std::cout << "Printing:     " << name << "... " << std::endl;

        PrintTensor <T, N> (*this, truncation_length, spacing);

        std::cout << "\n                                             ---***---                                             " << std::endl;
    };

    void PrintElements () const 
    {
        size_t length = dimensions [0];
        for (uint i = 1; i < N; i++)
        {
            length *= dimensions [i];
        };

        std::cout << std::endl;
        for (uint i = 0; i < length; i++)
        {
            std::cout << elements [i] << " ";
        };
        std::cout << std::endl;
    };

    #endif
};

// N = 1 template specialisation
template <typename T>
struct Tensor <T, 1>
{
    size_t length;
    T* elements;
    uint layer;

    Tensor () {};

    Tensor (const size_t dimensions [1], T e [])
    {
        layer = 0;
        size_t length = dimensions [0];

        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };
    };

    Tensor (const size_t dimensions [1], T* e, uint layer)
        : layer {layer}
    {
        length = dimensions [0];

        elements = e;
    };

    ~Tensor () 
    {
        if  (layer == 0)
        {
            delete [] elements;
        };
    };

    Tensor (const Tensor& t) = delete;

    const T& operator[] (uint idx) const 
    {
        return elements [idx];
    };

    T& operator[] (uint idx) 
    {
        return elements [idx];
    };

    T& index (uint idx)
    {
        return elements [idx];
    };
};

// TODO:
// template <typename T, size_t dim []>
// struct JaggedTensor
// {
//     Tensor <T, N>
// };