#pragma once
#include <chrono>

// CPU wall-clock timings, in milliseconds.
class PerformanceClock {
public:
    float restartMs() {
        const auto now = Clock::now();
        const float milliseconds =
            std::chrono::duration<float, std::milli>(now - start_).count();
        start_ = now;
        return milliseconds;
    }

    float elapsedMs() const {
        return std::chrono::duration<float, std::milli>(
            Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_ = Clock::now();
};
