#pragma once
#include <cmath>

typedef unsigned int uint;


template <typename T, size_t N>
struct Tensor
{
    size_t dimensions [N];
    T* elements;
    uint layer;

    Tensor <T, N - 1>** children;

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

    Tensor <T, N - 1>& operator[] (uint idx) 
    {
        return *(children [idx]);
    };

    const Tensor <T, N - 1>& operator[] (uint idx) const 
    {
        return *(children [idx]);
    };

    T index (uint indices [N])
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

    T index (uint idx)
    {
        return elements [idx];
    };
};

// TODO:
template <typename T, size_t dim []>
struct JaggedTensor
{
    Tensor <T, N>
};