#pragma once
#include <cmath>

#if DEBUG_LEVEL == 1
    #include <string>
    #include <random> 
#endif

typedef unsigned int uint;

typedef void (*Interim) (const uint, size_t);
template <typename T, typename DistributionType = std::uniform_real_distribution <T>>
struct Random
{
    std::random_device rd;
    std::mt19937 generator;
    DistributionType distribution;

    Random (          T mean = 0.0, T variance = 1.0) : generator (rd ()), distribution (mean, variance) {};
    Random (int seed, T mean = 0.0, T variance = 1.0) : generator (seed),  distribution (mean, variance) {};

    T number () 
    {
        return distribution (generator);
    };
};

void nullfn (const uint layer, size_t N) {};

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

//TODO: rewrite using loop instead of recursion?
template <typename FunctionType, typename InputType, typename OutputType, size_t N>
void __iteration 
(
    FunctionType f, 
    InputType x, 
    OutputType y, 
    size_t dimensions [N], 
    uint* const index, 
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

template <typename InputType, typename OutputType, size_t N>
void Iterate (void (*f) (const InputType, OutputType, uint* const), InputType x, OutputType y, size_t dimensions [N], Interim interim = nullfn)
{
    uint l = 0;
    uint* index = new uint [N]{};

    typedef void (*IterationFunction) (const InputType, OutputType, uint* const);

    __iteration <IterationFunction, InputType, OutputType, N> (f, x, y, dimensions, index, l, 0, interim);
};

template <typename InputType, typename OutputType, size_t M, size_t N>
void Iterate (void (*f) (const InputType, OutputType, uint* const), InputType x, OutputType y, size_t dimensions [M], uint* const outer_index, Interim interim = nullfn)
{
    uint l = 0;
    uint* index = new uint [M + N]{};

    for (uint i = 0; i < N; i++)
    {
        index [i] = outer_index [i];
    };

    typedef void (*IterationFunction) (const InputType, OutputType, uint* const);

    __iteration <IterationFunction, InputType, OutputType, M> (f, x, y, dimensions, index, l, N, interim);
};

void print_separator (const uint layer, const size_t N)
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
    std::ostream& os;
    size_t truncation_length;
    size_t spacing;

    PrintInput (const Tensor <T, N>& tensor, std::ostream& os, size_t truncation_length, size_t spacing)
        : tensor {tensor}, os {os}, truncation_length {truncation_length}, spacing {spacing}
    {};
};

template <typename T, size_t N>
struct PrintFunction
{
    typedef void (*f) (const PrintInput <T, N>&, void*, uint* const);
};

template <typename T, size_t N>
void PrintElement (const PrintInput <T, N>& input, void* output, uint* idx)
{
    const Tensor <T, N>& tensor = input.tensor;
    std::ostream& os = input.os;
    size_t truncation_length = input.truncation_length;
    size_t spacing = input.spacing;

    uint index [N];
    undo_alternating_sort (idx, index, N);
    
    // Crops each element (to 8 characters long) so display is uniform
    std::string character = std::to_string (tensor.index (index)).substr (0, truncation_length);
    os << character;
    
    for (uint i = 0; i < truncation_length - character.length () + spacing; i++)
    {
        os << " ";
    };
};

//TODO: Vary print formatting based on window size
template <typename T, size_t N>
void PrintTensor (const Tensor <T, N>& tensor, std::ostream& os = std::cout, size_t truncation_length = 8, size_t spacing = 2)
{
    PrintInput <T, N> input (tensor, os, truncation_length, spacing);

    size_t dim [N];
    alternating_sort (tensor.dimensions, dim, N);

    Iterate <const PrintInput <T, N>&, void*, N> (PrintElement, input, nullptr, dim, print_separator);
};

template <typename T>
void MatrixMultiply (const Tensor <T, 2>& V, const Tensor <T, 2>& W, Tensor <T, 2>& output)
{
    (V.dimensions [1] == W.dimensions [0]);
    (V.dimensions [0] == output.dimensions [0]);
    (W.dimensions [1] == output.dimensions [1]);

    for (uint i = 0; i < V.dimensions [0]; i++)
    {
        for (uint j = 0; j < W.dimensions [1]; j++)
        {
            for (uint k = 0; k < V.dimensions [1]; j++)
            {
                output [i][j] += V [i][k] *  W [k][j];
            };
        };
    };
};

template <typename T, size_t N>
struct Tensor
{
    size_t dimensions [N];
    size_t length = 0;
    uint layer = 0;
    T* elements = nullptr;
    
    Tensor <T, N - 1>** children = nullptr;

    #if DEBUG_LEVEL == 1

    size_t size = sizeof (Tensor <T, N>);
    const char* name = "default";

    #endif

    Tensor () {};

    // Parent Constructor
    Tensor (const size_t input_dimensions [N], const T e [])
    {
        Init (input_dimensions, e);
    };

    void Init (const size_t input_dimensions [N], const T e [])
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

        #if DEBUG_LEVEL == 1
        for (int i = 0; i < l; i++)
        {
            size += children [i] -> size;
        }; 
        #endif
    };

    // Child Constructor: shouldn't be called explicitly
    Tensor (const size_t input_dimensions [N], T* e, const uint layer)
        : layer {layer}
    {
        for (int i = 0; i < N; i++)
        {
            dimensions [i] = input_dimensions [i];
        };

        size_t l = dimensions [0];

        size_t dim [N - 1];
        length = 1;
        for (int i = 0; i < N - 1; i++)
        {
            dim [i] = dimensions [i + 1];
            length *= dimensions [i + 1];
        };

        children = new Tensor <T, N - 1>* [l];

        for (int i = 0; i < l; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (e + length * i), layer + 1);
        };

        elements = e; 

        #if DEBUG_LEVEL == 1
        for (int i = 0; i < l; i++)
        {
            size += children [i] -> size;
        }; 
        #endif
    };

    Tensor (const size_t input_dimensions [N])
    {
        Init (input_dimensions);
    };

    void Init (const size_t input_dimensions [N])
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

        #if DEBUG_LEVEL == 1
        for (int i = 0; i < l; i++)
        {
            size += children [i] -> size;
        };
        #endif 
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

        // Child tensors do not own the elements resource, only reference it
        // elements pointers are deleted as they go out of scope, and then
        // the memory itself is deleted [] by the parent tensor
        if  (layer == 0 && elements != nullptr)
        {
            delete [] elements;
        };

        //TODO: warn if destroying uninitialised tensor
    };

    // Copy Constructor
    Tensor (const Tensor& t) = delete;

    // Move Constructor
    Tensor (Tensor&& t) noexcept
    {
        Init (static_cast <Tensor&&> (t));
        // Init (std::move (t));
    };

    void Init (Tensor&& t) noexcept
    {
        // std::cout << "Moved Tensor" << std::endl;
        
        elements = t.elements;
        t.elements = nullptr;
        
        children = t.children;
        t.children = nullptr;
        
        for (uint i = 0; i < N; i++)
        {
            dimensions [i] = t.dimensions [i];
            t.dimensions [i] = 0;
        };

        layer = t.layer;

        length = t.length;
        t.length = 0;

        #if DEBUG_LEVEL == 1

        size = t.size;
        t.size = 0;

        name = t.name;
        t.name = nullptr;

        #endif
    };

    T& index (const uint indices [N])
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

    const T& index (const uint indices [N]) const
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

    Tensor <T, N - 1>& operator[] (const uint idx) 
    {
        return *(children [idx]);
    };

    const Tensor <T, N - 1>& operator[] (const uint idx) const 
    {
        return *(children [idx]);
    };

    T& operator[] (const uint indices [N])
    {
        return index (indices);
    };

    const T& operator[] (const uint indices [N]) const 
    {
        return index (indices);
    };

    Tensor <T, N - 1>& operator[] (const int idx) 
    {
        return *(children [idx]);
    };

    const Tensor <T, N - 1>& operator[] (const int idx) const 
    {
        return *(children [idx]);
    };

    T& operator[] (const int indices [N])
    {
        return index (indices);
    };

    const T& operator[] (const int indices [N]) const 
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

    void SetElements (const T input [], const size_t input_length)
    {
        if (input_length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input [i];
            };
        };
    };

    void SetElements (const T input)
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = input;
        };
    };

    void SetElements (T (*Generator) (uint))
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = Generator (i);
        };
    };

    void Rotate ()
    {
        T rotated_elements [length];

        size_t spacing = dimensions [N - 1] * dimensions [N - 2];

        for (uint i = 0; i < length / spacing; i++)
        {
            for (uint j = 0; j < spacing; j++)
            {
                rotated_elements [i * spacing + j] = elements [(i + 1) * spacing - 1 - j];
            };
        };

        for (int i = 0; i < length; i++)
        {
            elements [i] = rotated_elements [i];
        };
    };

    void Flip ()
    {
        T flipped_elements [length];
        for (uint i = 0; i < length; i++)
        {
            flipped_elements [i] = elements [length - 1 - i];
        };

        size_t flipped_dimensions [N];
        for (uint i = 0; i < N; i++)
        {
            flipped_dimensions [i] = dimensions [N - 1 - i];
        };

        layer = 0;

        for (int i = 0; i < N; i++)
        {
            dimensions [i] = flipped_dimensions [i];
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

        for (int i = 0; i < length; i++)
        {
            elements [i] = flipped_elements [i];
        };

        size_t l = dimensions [0];

        // If creation fails, children won't be left dangling
        Tensor <T, N - 1>** temporary = new Tensor <T, N - 1>* [l];

        if (children != nullptr)
        {
            for (size_t i = 0; i < dimensions [N - 1]; i++)
            {
                delete children [i];
            };
            delete [] children;
        };

        children = temporary;

        for (int i = 0; i < l; i++)
        {
            children [i] = new Tensor <T, N - 1> (dim, (flipped_elements + separation * i), 1);
        };

        #if DEBUG_LEVEL == 1
        size = sizeof (Tensor <T, N>);
        for (uint i = 0; i < l; i++)
        {
            size += children [i] -> size;
        };
        #endif
    };

    const Tensor <T, N> Copy () const
    {
        return Tensor (dimensions, elements);
    };

    template <typename DistributionType = std::uniform_real_distribution <T>>
    void Randomise (T mean, T variance)
    {
        Random <T, DistributionType> r (SEED, mean, variance);

        for (uint i = 0; i < length; i++)
        {
            elements [i] = r.number ();
        };
    };

    template <typename DistributionType = std::uniform_real_distribution <T>>
    void Randomise ()
    {
        Random <T, DistributionType> r (SEED, 0.0, float(1.0) / float(length));

        for (uint i = 0; i < length; i++)
        {
            elements [i] = r.number ();
        };
    };

    #if DEBUG_LEVEL == 1

    void SetElements (Random <T>* r)
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = r -> number ();
        };
    };

    friend std::ostream& operator<<(std::ostream& os, const Tensor <T, N>& tensor)
    {
        tensor.Print (nullptr, os);
        return os;
    };

    void Print (const char* printname = nullptr, std::ostream& os = std::cout, size_t truncation_length = 8) const
    {
        size_t spacing = 2;

        if (printname != nullptr)
            os << "Printing:     " << printname << "... " << std::endl;
        else if (name != nullptr)
            os << "Printing:     " << name << "... " << std::endl;

        PrintTensor <T, N> (*this, os, truncation_length, spacing);

        os << "\n                                             ---***---                                             " << std::endl;
    };

    void PrintElements (std::ostream& os = std::cout) const 
    {
        size_t length = dimensions [0];
        for (uint i = 1; i < N; i++)
        {
            length *= dimensions [i];
        };

        os << std::endl;
        for (uint i = 0; i < length; i++)
        {
            os << elements [i] << " ";
        };
        os << std::endl;
    };


    #endif
};

// N = 1 template specialisation
template <typename T>
struct Tensor <T, 1>
{
    size_t dimensions [1];
    size_t length = 0;
    uint layer = 0;
    T* elements = nullptr;

    #if DEBUG_LEVEL == 1

    size_t size = sizeof (Tensor <T, 1>);
    const char* name = "default";

    #endif

    Tensor () {};

    // Parent Constructor
    Tensor (const size_t dimension, T e [])
    {
        Init (dimension, e);
    };

    Tensor (const size_t input_dimensions [1], T e [])
    {
        Init (input_dimensions [0], e);
    };

    void Init (const size_t dimension, T e [])
    {
        layer = 0;
        length = dimension;
        dimensions [0] = length;

        elements = new T [length];
        for (int i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };

        #if DEBUG_LEVEL == 1
        size += sizeof (T) * length;  // elements
        #endif
    };

    // Child Constructor, shouldn't be called explicitly
    Tensor (const size_t input_dimensions [1], T* e, const uint layer)
        : layer {layer}
    {
        length = input_dimensions [0];
        dimensions [0] = length;

        elements = e;

        #if DEBUG_LEVEL == 1
        size += sizeof (T) * length;  // elements
        #endif
    };

    Tensor (const size_t dimension)
    {
        Init (dimension);
    };

    Tensor (const size_t dimensions [1])
    {
        Init (dimensions [0]);
    };

    void Init (const size_t dimension)
    {
        layer = 0;
        length = dimension;
        
        dimensions [0] = length;
        elements = new T [length] {};

        #if DEBUG_LEVEL == 1
        size += sizeof (T) * length;  // elements
        #endif
    };

    ~Tensor () 
    {
        if  (layer == 0 && elements != nullptr)
        {
            delete [] elements;
        };

        //TODO: warn if destroying uninitialised tensor
    };

    // Copy constructor
    Tensor (const Tensor& t) = delete;

    // Move Constructor
    Tensor (Tensor&& t) noexcept
    {
        Init (static_cast <Tensor&&> (t));
    };

    void Init (Tensor&& t) noexcept
    {
        // std::cout << "Moved Tensor <T, 1>" << std::endl;
        elements = t.elements;
        t.elements = nullptr;

        layer = t.layer;

        length = t.length;
        dimensions [0] = length;
        t.length = 0;

        #if DEBUG_LEVEL == 1

        size = t.size;
        t.size = 0;

        name = t.name;
        t.name = nullptr;

        #endif
    };

    const T& operator[] (const uint idx) const 
    {
        return elements [idx];
    };

    T& operator[] (const uint idx) 
    {
        return elements [idx];
    };
    
    const T& index (const uint idx) const
    {
        return elements [idx];
    };

    T& index (const uint idx)
    {
        return elements [idx];
    };

    const T& index (const uint indices [1]) const
    {
        return elements [indices [0]];
    };

    T& index (const uint indices [1])
    {
        return elements [indices [0]];
    };

    void SetElements (const T input)
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = input;
        };
    };

    void SetElements (T (*Generator) (uint))
    {
        for (uint i = 0; i < length; i++)
        {
            elements [i] = Generator (i);
        };
    };

    void SetElements (const T input [], const size_t input_length)
    {
        if (input_length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input [i];
            };
        };
    };

    void SetElements (const Tensor <T, 1>& input)
    {
        if (input.length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input.elements [i];
            };
        };
    };

    void SetElements (const Tensor <T, 1>* input)
    {
        if (input -> length == length) 
        {
            for (uint i = 0; i < length; i++)
            {
                elements [i] = input -> elements [i];
            };
        };
    };

    //TODO: this seems like a feature that encourages bad variable management
    void Reshape (const size_t input_dimensions [1], T e [])
    {
        length = input_dimensions [0];
        dimensions [0] = length;

        delete [] elements;
        elements = new T [length];

        for (uint i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };
    };

    void Reshape (const size_t dimension, T e [])
    {
        length = dimension;
        dimensions [0] = length;

        delete [] elements;
        elements = new T [length];

        for (uint i = 0; i < length; i++)
        {
            elements [i] = e [i];
        };
    };

    void Reshape (const size_t input_dimensions [1])
    {
        length = input_dimensions [0];
        dimensions [0] = length;

        delete [] elements;
        elements = new T [length]{};
    };

    void Reshape (const size_t dimension)
    {
        length = dimension;
        dimensions [0] = length;

        delete [] elements;
        elements = new T [length]{};
    };


    template <typename DistributionType = std::uniform_real_distribution <T>>
    void Randomise (T mean, T variance)
    {
        Random <T, DistributionType> r (SEED, mean, variance);

        for (uint i = 0; i < length; i++)
        {
            elements [i] = r.number ();
        };
    };

    template <typename DistributionType = std::uniform_real_distribution <T>>
    void Randomise ()
    {
        Random <T, DistributionType> r (SEED, 0.0, float(1.0) / float(length));

        for (uint i = 0; i < length; i++)
        {
            elements [i] = r.number ();
        };
    };

    #if DEBUG_LEVEL == 1

    friend std::ostream& operator<<(std::ostream& os, const Tensor <T, 1>& tensor)
    {
        tensor.Print (nullptr, os);
        return os;
    };

    void Print (const char* printname = nullptr, std::ostream& os = std::cout, size_t truncation_length = 8) const
    {
        size_t spacing = 2;

        if (printname != nullptr)
            os << "Printing:     " << printname << "... " << std::endl;
        else if (name != nullptr)
            os << "Printing:     " << name << "... " << std::endl;

        PrintTensor <T, 1> (*this, os, truncation_length, spacing);

        os << "\n                                             ---***---                                             " << std::endl;
    };

    #endif
};