#pragma once

namespace meta
{
    namespace detail
    {
        // std::remove_reference, std::remove_cv
        template <typename T> struct remove_qualifier               { using type = T; };
        template <typename T> struct remove_qualifier <T&>          { using type = T; };
        template <typename T> struct remove_qualifier <T&&>         { using type = T; };
        template <typename T> struct remove_qualifier <const T>     { using type = T; };
        template <typename T> struct remove_qualifier <const T&>    { using type = T; };
        template <typename T> struct remove_qualifier <volatile T>  { using type = T; };
        template <typename T> struct remove_qualifier <volatile T&> { using type = T; };
        
        // std:: is_same_type
        template <typename T1, 
                  typename T2> struct same_type        { static constexpr bool value = false; };
        template <typename T>  struct same_type <T, T> { static constexpr bool value = true;  };
        
        // std::conditional
        template <bool, typename T, typename F> struct conditional               { typedef T type; };
        template       <typename T, typename F> struct conditional <false, T, F> { typedef F type; };

        template <bool, typename = void> struct SFINAE {};

        template <typename T>
        struct SFINAE <true, T>
        {
            using type = T;
        };
    };

    
    template <typename T>
    using unqualified = detail::remove_qualifier <T>::type;
    
    template <typename T1, typename T2> constexpr bool same_type      =  detail::same_type <unqualified <T1>, unqualified <T2>>::value;
    template <typename T1, typename T2> constexpr bool different_type = !detail::same_type <unqualified <T1>, unqualified <T2>>::value;
    
    template <bool B, typename T, typename F> 
    using conditional = detail::conditional <B, T, F>::type;

    template <bool B, typename T = void>
    using SFINAE = detail::SFINAE <B, T>::type;
};
