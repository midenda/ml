#pragma once
#include <cmath>

typedef unsigned int uint;

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
    uint layer;

    typedef typename conditional <(N > 1), Tensor <T, N - 1>, void> ::type ChildType;
    ChildType** children;

    Tensor () {};

    // Parent Constructor
    Tensor (size_t input_dimensions [N], T e [])
    {
        layer = 0;

        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        size_t length = dimensions [0];
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
    Tensor (size_t input_dimensions [N], T* e, uint layer)
        : layer {layer}
    {
        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        size_t length = dimensions [0];

        size_t dim [N - 1];
        size_t separation = 1;
        for (int i = 0; i < N - 1; i++)
        {
            dim [i] = dimensions [i + 1];
            separation *= dimensions [i + 1];
        };

        children = new Tensor <T, N - 1>* [length];

        for (int i = 0; i < length; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (e + separation * i), layer + 1);
        };

        elements = e; 
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

    ChildType& operator[] (size_t idx) 
    {
        return *(children [idx]);
    };

    const ChildType& operator[] (size_t idx) const 
    {
        return *(children [idx]);
    };
};

// N = 1 template specialisation
template <typename T>
struct Tensor <T, 1>
{
    size_t length;
    T* elements;
    uint layer;

    Tensor () {};

    Tensor (size_t dimensions [1], T e [])
    {
        layer = 0;
        size_t length = dimensions [0];

        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };
    };

    Tensor (size_t dimensions [1], T* e, uint layer)
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

    T operator[] (size_t idx) 
    {
        return elements [idx];
    };

    const T operator[] (size_t idx) const 
    {
        return elements [idx];
    };
};