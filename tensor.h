#pragma once
#include <cmath>

// Implementation of std::conditional
template <bool, typename T, typename F>
struct conditional
{
    typedef T type;
};

template <typename T, typename F>
struct conditional <false, T, F>
{
    typedef F type;
};


template <typename T, size_t N>
struct Tensor
{
    size_t dimensions [N];
    T* elements;
    int start = 0;

    typedef typename conditional <(N > 1), Tensor <T, N - 1>, void >::type ChildType;
    ChildType* children;

    Tensor () {};

    // Parent Constructor
    Tensor (size_t dimensions [N], T e [], size_t length)
    {
        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };

        size_t dim [N - 1];
        size_t separation = 1;
        for (int i = 0; i < N - 1; i++)
        {
            dim [i] = dimensions [i + 1];
            separation *= dimensions [i + 1];
        };

        size_t l = dimensions [0];
        children = new Tensor <T, N - 1> [l];

        for (int i = 0; i < l; i++)
        {
            children [i] = Tensor <T, N - 1> (dim, (elements + separation * i));
        };
    };

    // Child Constructor: shouldn't be called explicitly
    Tensor (size_t dimensions [N], T* e)
    {
        size_t length = dimensions [0];

        size_t dim [N - 1];
        size_t separation = 1;
        for (int i = 0; i < N - 1; i++)
        {
            dim [i] = dimensions [i + 1];
            separation *= dimensions [i + 1];
        };

        children = new Tensor <T, N - 1> [length];

        for (int i = 0; i < length; i++)
        {
            children [i] = Tensor <T, N - 1> (dim, (e + separation * i));
        };

        elements = e; 
    };

    const Tensor <T, N>& operator() (size_t dimensions [], T elements []) const
    {
        return Tensor <T, N> (dimensions, elements);
    };

    ChildType& operator[] (size_t idx) 
    {
        return children [idx];
    };

    const ChildType& operator[] (size_t idx) const 
    {
        return children [idx];
    };
};

// N = 1 template specialisation
template <typename T>
struct Tensor <T, 1>
{
    size_t length;
    T* elements;
    void* children; // TODO: is this necessary?

    Tensor () {};

    Tensor (size_t dimensions [1], T e [], size_t length)
        : length {length}
    {
        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };

        children = nullptr;
    };

    Tensor (size_t dimensions [1], T* e)
    {
        length = dimensions [0];

        elements = e;
        children = nullptr;
    };

    const Tensor <T, 1>& operator() (size_t dim [], T e []) const
    {
        return Tensor <T, 1> (dim, e);
    };

    T operator[] (size_t idx) 
    {
        return elements [idx];
    };

    const T operator[] (size_t idx) const 
    {
        return elements [idx];
    };
};

// int main ()
// {
//     size_t dimensions [3] = {2, 3, 2};
//     float elements [] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     Tensor <float, 3> T (dimensions, elements, 12);

//     {
//         for (int i = 0; i < 2; i++)
//             for (int j = 0; j < 3; j++)
//                 for (int k = 0; k < 2; k++)
//                     std::cout << T [i][j][k] << std::endl;
//     };
// };
