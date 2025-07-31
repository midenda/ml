#include <string>

#include "../tuple.h"

// TODO: i don't like this functor syntax
//? can't use static operator()
struct PrintTuple 
{
    template <typename T, typename... Args>
    static void call (T input, Args... args)
    {
        std::cout << input << std::endl;
    };
};

template <typename T>
using printtype = void (*) (T);

void test_tuple ()
{
    Tuple <double, bool, int, const char*> T { 3.1, true, 4, "string"};
    Tuple other {(float)3.1, (bool)true, (int)4, (const char*)"string"};
    Tuple third { 3.1, true, 4, "string" };

    std::cout << T.Get <0> () << std::endl;
    std::cout << T.Get <1> () << std::endl;
    std::cout << T.Get <2> () << std::endl;
    std::cout << T.Get <3> () << std::endl;

    RepeatTuple <1, int, bool, double, const char*> repeat  { 4, true, 3.4, "repeats once!" };
    RepeatTuple <4, int, bool, double, const char*> repeat4 { 2, false, 2.8, "repeats 4 times!" };


    // T.Call <PrintTuple> ();
    // other.Call <PrintTuple> ();
    // third.Call <PrintTuple> ();

    repeat.Print ();
    repeat4.Print ();

    // auto [a, b, c] = T; //TODO: implement std::tuple_size for structured binding assignment

    [[ maybe_unused ]]
    Tuple empty {};
    // empty.Call <PrintTuple> (); // Error
};

