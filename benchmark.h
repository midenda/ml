#pragma once

#include <iostream>
#include <chrono>

#define PROFILING 1

#ifdef PROFILING
    #define ProfileScope(name) Timer timer##__line__(name)
    #define Profile() ProfileScope(__FUNCTION__)
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

    void __session (const char* filepath)
    {
        if (active_session) 
            __end_session ();

        active_session = true;

        stream.open (filepath);
        stream << "{\"otherData\": {},\"traceEvents\":[";
    };

    void __end_session () 
    {
        if (!active_session) return;

        active_session = false;

        stream << "]}";
        stream.close ();

        count = 0;
    };

    void __write_profile (InstrumentorProfile result) 
    {
        if (count++ > 0)
            stream << ",";

        stream << "{";
        stream << "\"cat\":\"function\",";
        stream << "\"dur\":" << result.duration << ",";
        stream << "\"name\":\"" << result.name << "\",";
        stream << "\"ph\":\"X\",";
        stream << "\"pid\":0,";
        stream << "\"tid\":0,";
        stream << "\"ts\":" << result.start;
        stream << "}";
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

    static void Session (const char* filepath = "./profile.json")
    {
        Get ().__session (filepath);
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

        InstrumentorProfile result (name, start.time_since_epoch ().count (), duration);

        Instrumentor::WriteProfile (result);

        running = false;
    };
};