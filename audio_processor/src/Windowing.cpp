#include "Windowing.h"
#include <cmath>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// Create Hamming window
std::vector<double> Windowing::createHammingWindow(int frame_duration) {
    std::vector<double> hamming(frame_duration);
    for (int i = 0; i < frame_duration; ++i) {
        hamming[i] = 0.54 - 0.46 * std::cos(2 * M_PI * i / (frame_duration - 1));
    }
    return hamming;
}

// Apply Hamming window
void Windowing::applyHammingWindow(std::vector<double>& frame, const std::vector<double>& hamming_window) {
    for (size_t i = 0; i < frame.size(); ++i) {
        frame[i] *= hamming_window[i];
    }
}
