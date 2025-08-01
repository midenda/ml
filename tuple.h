#pragma once

#if DEBUG_LEVEL == 1
    #define FUNC_SIG std::cout << __PRETTY_FUNCTION__ << std::endl;

    #include <string>
#endif

#include "./meta.h"

//TODO: mark constexpr functions as constexpr

typedef unsigned int uint;

template <uint I, typename T>
struct TupleLeaf
{
    T value;

    TupleLeaf () {};
    
    TupleLeaf (T&& value)
        : value { static_cast <T&&> (value) } {};

    template <typename... Args>
    TupleLeaf (Args... args) 
        : value { args... } {};

    //? Init ?

    #if DEBUG_LEVEL == 1

    void Print (std::ostream& os)
    {
        os << value << std::endl;
    };

    #endif
};

// empty tuple element
template <uint I>
struct TupleLeaf <I, void>
{};

template <typename C, typename T>
concept Callable = requires (C, T t) 
{ 
    C :: template call <T> (t); 
};

template <typename C, typename F, typename T> //TODO: concept checking for foreach, propagate, backpropagate
concept Callable2 = requires (C c, F f, T t) 
{ 
    (c.*f) (t); 
};

//TODO: refactor
template <typename... Ts>
class MakeTuple
{
private:

    template <uint I, typename _, typename... Types> 
    struct GetType 
    {
        using type = GetType <I - 1, Types...>::type;
    };

    template <typename T, typename... Types>
    struct GetType <(uint)0, T, Types...>
    {
        using type = T;
    };

    
    template <typename T, typename ReturnType, typename... Args>
    using FunctionReference = ReturnType (T::*) (Args...);
    
    template <uint... integers>
    struct Index 
    { 
        struct Tuple
        : TupleLeaf <integers, Ts>...
        {
            static constexpr const uint N = sizeof... (Ts);

            template <uint I = 0>
            using Type = GetType <I, Ts...>::type; 

            Tuple () 
                : TupleLeaf <integers, Ts> {}... {};
            
            Tuple (Ts&&... args) 
                : TupleLeaf <integers, Ts> { static_cast <Ts&&> (args) }... {}; //? can this be done with copy elision instead of move semantics?

            //! using the Tuple (Ts&&... args) constructor with an incorrect number of arguments instead calls this constructor
            //? this constructor requires all elements to accept the same arguments as inputs
            //? possibly impose some kind of type checking concept
            //? eg requires (std::is_base_of_v <BaseType, Ts> && ...)
            //? only allow this constructor for TypedTuple ?
            template <typename... Args>
            Tuple (Args... args)
                : TupleLeaf <integers, Ts> { args... }... {};

            template <typename... Args>
            void Init (Args... args)
            {
                (static_cast <TupleLeaf <integers, Ts>*> (this) -> value.Init (args...), ...);
            };

            void Init (Ts&&... args)
            {
                (static_cast <TupleLeaf <integers, Ts>*> (this) -> value.Init (static_cast <Ts&&> (args)), ...);
            };

            Tuple (const Tuple& t) = delete;
            Tuple (Tuple&& t)      = delete;

            template <uint I>
            Type <I>& Get () 
            {
                return static_cast <TupleLeaf <I, Type <I>>*> (this) -> value;
            };

            template <class Class, typename OutputType, typename... InputTypes> 
            void ForEach (OutputType (Class::* F) (InputTypes...), InputTypes... inputs) 
            {
                ((Get <integers> ().*F)(inputs...), ...);
            };
            
            template <uint I = 0, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args>
            meta::SFINAE <I < N, OutputType> Propagate (OutputType (Class::* F) (InputType, Parameters...), InputType&& input, Args&&... args)
            {
                return Propagate <I + 1> (F, (Get <I> ().*F)(static_cast <InputType&&> (input), static_cast <Args&&> (args)...), static_cast <Args&&> (args)...);
            };
            template <uint I, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args> 
            meta::SFINAE <I == N, OutputType> Propagate (OutputType (Class::*) (InputType, Parameters...), InputType&& input, Args&&...) 
            {
                return static_cast <InputType&&> (input);
            };
            
            template <uint I = N - 1, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args>
            meta::SFINAE <0 < I, OutputType> BackPropagate (OutputType (Class::* F) (InputType, Parameters...), InputType (Class::* map) (), InputType&& input, Args&&... args)
            {
                (Get <I> ().*F)((Get <I - 1> ().*map) (), static_cast <Args&&> (args)...);
                return BackPropagate <I - 1> (F, map, static_cast <InputType&&> (input), static_cast <Args&&> (args)...);
            };
            template <uint I = N - 1, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args>
            meta::SFINAE <I == 0, OutputType> BackPropagate (OutputType (Class::* F) (InputType, Parameters...), InputType (Class::*) (), InputType&& input, Args&&... args) 
            {
                return (Get <0> ().*F) (static_cast <InputType&&> (input), static_cast <Args&&> (args)...);
            };

            #if DEBUG_LEVEL == 1

            template <uint I = 0>
            meta::SFINAE <I < N, void> Print (std::ostream& os = std::cout)
            {
                static_cast <TupleLeaf <I, Type <I>>*> (this) -> Print (os);
                Print <I + 1> (os);
            };
            template <uint I>
            meta::SFINAE <I == N, void> Print (std::ostream& = std::cout) {};

            #endif

            //TODO: only works if all elements are same type, can't pass templates
            //? use functor (see Call() below, can't template operator() so messy syntax) or ...?
            // template <uint I = 0, typename Function, typename... Args> 
            // meta::SFINAE <I < N> foreach (Function F, Args... args) 
            // { 
            //     F (Get <I> (), args...);
            //     foreach <I + 1, Function, Args...> (F, args...);
            // };
            // template <uint I, typename Function, typename... Args> 
            // meta::SFINAE <I == N> foreach (Function, Args...) {};

            //! deprecate
            // template <typename Functor, typename... Args> requires Callable <Functor, Type <0>>
            // [[ deprecated ("functionality moved to 'foreach'") ]]
            // meta::SFINAE <0 < N> Call (Args... args)
            // {
            //     Functor :: template call <Type <0>> (Get <0> (), args...);
            //     Call <1, Functor, Args...> (args...); 
            // };
            // template <uint I = 0, typename Functor, typename... Args> requires Callable <Functor, Type <I>> //? default argument I = 0 doesn't allow Call <Functor> ()
            // meta::SFINAE <I < N> Call (Args... args)
            // {
            //     Functor :: template call <Type <I>> (Get <I> (), args...);
            //     Call <I + 1, Functor, Args...> (args...);
            // };
            // template <uint I = 0, typename Functor, typename... Args> meta::SFINAE <I == N> Call (Args...) {};
        };
    };

    template <uint index, uint... integers>
    struct Range
    {
        using Tuple = typename MakeTuple::Range <index - 1, index, integers...>::Tuple;
    };

    template <uint... integers>
    struct Range <0, integers...>
    {
        using Tuple = typename MakeTuple::Index <0, integers...>::Tuple;
    };

public:
    using Tuple = Range <sizeof... (Ts) - 1> :: Tuple;
};

template <size_t N, typename... Ts>
struct MakeRepeatTuple
{
private:
    template <uint I, typename... All>
    struct Sum 
    {
        struct Tuple : Sum <I - 1, All..., Ts...>::Tuple
        {
            // Inherit constructors
            using Sum <I - 1, All..., Ts...>::Tuple::Tuple;

            // Duplicate arguments to instantiate repeat elements
            Tuple (Ts... args) requires (I - 1 > 0)
                : Sum <I - 1, All..., Ts...>:: Tuple (args..., args..., args...) {};

            Tuple (All... all, Ts... args) requires (I - 1 > 0)
                : Sum <I - 1, All..., Ts...>:: Tuple (all..., args..., args...) {};

            Tuple (All... all, Ts... args) requires (I - 1 == 0)
                : Sum <I - 1, All..., Ts...>:: Tuple (static_cast <All&&> (all)..., static_cast <Ts&& >(args) ...) {};
        };
    };

    template <typename... All>
    struct Sum <0, All...>
    {
        using Tuple = MakeTuple <All...> :: Tuple;
    };

public:
    using Tuple = Sum <N - 1, Ts...>::Tuple;
};

template <typename... Ts>
struct MakeRepeatTuple <1, Ts...>
{
    // Inherit constructors
    using Tuple = MakeTuple <Ts...> :: Tuple;
};

template <size_t N, typename... Ts>
struct RepeatTuple : MakeRepeatTuple <N, Ts...> :: Tuple
{
    // Inherit constructors
    using MakeRepeatTuple <N, Ts...> :: Tuple::Tuple;
};

// Empty Base Optimisation
template <size_t N>
struct RepeatTuple <N> {};

template <typename... Ts>
struct Tuple : MakeTuple <Ts...>::Tuple 
{
    // Inherit constructors
    using MakeTuple <Ts...>::Tuple::Tuple;
}; 

// Empty Base Optimisation
template <>
struct Tuple <> {};

// class template argument deduction guide to allow "Tuple t { 3.1, true, 4 };" syntax
template <typename... Ts>
Tuple (Ts...) -> Tuple <Ts...>; 