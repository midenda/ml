#pragma once

#include <iostream>
#include <chrono>
#include <regex>

#define PROFILING 1

#ifdef PROFILING
    #define ProfileScope(name) Timer timer##__LINE__(name)
    #define Profile ProfileScope(__FUNCTION__)
    #define BenchmarkSession Instrumentor::Session (__FILE__);
#else
    #define ProfileScope(name)
#endif

struct InstrumentorProfile
{
    const char* name;
    long long start;
    long long duration;

    InstrumentorProfile (const char* name, long long start, long long duration)
        : name {name}, start {start}, duration {duration}
    {};
};

class Instrumentor
{
private:
    bool active_session = false;
    std::ofstream stream;
    unsigned int count;

    Instrumentor () {};

    void __session (const char* filename)
    {
        std::string filepath = std::regex_replace (filename, std::regex ("\\."), "-");

        filepath = "./profiles/" + filepath + ".json";

        if (active_session) 
            __end_session ();

        active_session = true;

        stream.open (filepath);
        stream << "{\n\t\"otherData\": {},\n\t\"traceEvents\": \n\t\t[";
    };

    void __end_session () 
    {
        if (!active_session) return;

        active_session = false;

        stream << "\n\t\t]\n}";
        stream.close ();

        count = 0;
    };

    void __write_profile (InstrumentorProfile result) 
    {
        if (count++ > 0)
            stream << ",";

        stream << "\n\t\t\t{";
        stream << "\n\t\t\t\t\"cat\":\"function\",";
        stream << "\n\t\t\t\t\"dur\":" << result.duration << ",";
        stream << "\n\t\t\t\t\"name\":\"" << result.name << "\",";
        stream << "\n\t\t\t\t\"ph\":\"X\",";
        stream << "\n\t\t\t\t\"pid\":0,";
        stream << "\n\t\t\t\t\"tid\":0,";
        stream << "\n\t\t\t\t\"ts\":" << result.start;
        stream << "\n\t\t\t}";
    };

public:
    ~Instrumentor ()
    {
        __end_session ();
    };

    static Instrumentor& Get ()
    {
        static Instrumentor instance;
        return instance;
    };

    // Deleted copy constructor
    Instrumentor (const Instrumentor&) = delete;

    static void Session (const char* filename)
    {
        Get ().__session (filename);
    };

    static void WriteProfile (InstrumentorProfile result)
    {
        Get ().__write_profile (result);
    };
};

class Timer
{
private:
    const char* name;
    std::chrono::time_point <std::chrono::steady_clock> start;
    bool running = true;

public:
    Timer (const char* name)
        : name {name}
    {
        start = std::chrono::steady_clock::now ();
    };

    ~Timer ()
    {
        if (running)
            Stop ();
    };

    void Stop ()
    {   
        std::chrono::time_point <std::chrono::steady_clock> end = std::chrono::steady_clock::now ();

        long long duration = std::chrono::duration_cast <std::chrono::microseconds> (end - start).count ();
        long long start_time = std::chrono::duration_cast <std::chrono::microseconds> (start.time_since_epoch ()).count ();

        InstrumentorProfile result (name, start_time, duration);

        Instrumentor::WriteProfile (result);

        running = false;
    };
};