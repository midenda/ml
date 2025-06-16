#pragma once

#if DEBUG_LEVEL == 1
    #define FUNC_SIG std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

//TODO: mark constexpr functions as constexpr

typedef unsigned int uint;

template <bool, typename = void> struct SFINAE {};

template <typename T>
struct SFINAE <true, T>
{
    using type = T;
};

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

            Tuple () {};
            
            Tuple (Ts&&... args) 
                : TupleLeaf <integers, Ts> { static_cast <Ts&&> (args) }... {}; //? can this be done with copy elision instead of move semantics?

            template <typename... Args>
            Tuple (Args... args)
                : TupleLeaf <integers, Ts> { args... }... {};

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
            typename SFINAE <I < N, OutputType>::type Propagate (OutputType (Class::* F) (InputType, Parameters...), InputType&& input, Args&&... args)
            {
                return Propagate <I + 1> (F, (Get <I> ().*F)(static_cast <InputType&&> (input), static_cast <Args&&> (args)...), static_cast <Args&&> (args)...);
            };
            template <uint I, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args> 
            typename SFINAE <I == N, OutputType>::type Propagate (OutputType (Class::*) (InputType, Parameters...), InputType&& input, Args&&...) 
            {
                return static_cast <InputType&&> (input);
            };
            
            template <uint I = N, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args>
            typename SFINAE <0 < I, OutputType>::type BackPropagate (OutputType (Class::* F) (InputType, Parameters...), InputType&& input, Args&&... args)
            {
                return BackPropagate <I - 1> (F, (Get <I - 1> ().*F)(static_cast <InputType&&> (input), static_cast <Args&&> (args)...), static_cast <Args&&> (args)...);
            };
            template <uint I, typename OutputType, class Class, typename InputType, typename... Parameters, typename... Args>
            typename SFINAE <I == 0, OutputType>::type BackPropagate (OutputType (Class::*) (InputType, Parameters...), InputType&& input, Args&&... args) 
            {
                return static_cast <InputType&&> (input);
            };

            //TODO: only works if all elements are same type, can't pass templates
            //? use functor (see Call() below, can't template operator() so messy syntax) or ...?
            // template <uint I = 0, typename Function, typename... Args> 
            // typename SFINAE <I < N>::type foreach (Function F, Args... args) 
            // { 
            //     F (Get <I> (), args...);
            //     foreach <I + 1, Function, Args...> (F, args...);
            // };
            // template <uint I, typename Function, typename... Args> 
            // typename SFINAE <I == N>::type foreach (Function, Args...) {};

            //! deprecate
            // template <typename Functor, typename... Args> requires Callable <Functor, Type <0>>
            // [[ deprecated ("functionality moved to 'foreach'") ]]
            // typename SFINAE <0 < N>::type Call (Args... args)
            // {
            //     Functor :: template call <Type <0>> (Get <0> (), args...);
            //     Call <1, Functor, Args...> (args...); 
            // };ß
            // template <uint I = 0, typename Functor, typename... Args> requires Callable <Functor, Type <I>> //? default argument I = 0 doesn't allow Call <Functor> ()
            // typename SFINAE <I < N>::type Call (Args... args)
            // {
            //     Functor :: template call <Type <I>> (Get <I> (), args...);
            //     Call <I + 1, Functor, Args...> (args...);
            // };
            // template <uint I = 0, typename Functor, typename... Args> typename SFINAE <I == N>::type Call (Args...) {};
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

template <typename... Ts>
struct Tuple : MakeTuple <Ts...>::Tuple 
{
    // Tuple (Ts... args) 
    //     : MakeTuple <Ts...>::Tuple (args...) {};
    
    Tuple (Ts&&... args) 
        : MakeTuple <Ts...>::Tuple (static_cast <Ts&&> (args)...) {};

    template <typename... Args>
    Tuple (Args... args) 
        : MakeTuple <Ts...>::Tuple (args...) {};
}; 

// empty base optimisation
template <>
struct Tuple <> {};

// class template argument deduction guide to allow "Tuple t { 3.1, true, 4 };" syntax
template <typename... Ts>
Tuple (Ts...) -> Tuple <Ts...>; 