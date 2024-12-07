#include "Windowing.h"
#include <arm_neon.h>
#include <cmath>

// Create window functions

// Create Hamming window
std::vector<double> Windowing::createHammingWindow(int frame_duration) {
    std::vector<double> hamming(frame_duration);
    for (int i = 0; i < frame_duration; ++i) {
        hamming[i] = 0.54 - 0.46 * std::cos(2 * M_PI * i / (frame_duration - 1));
    }
    return hamming;
}

// Apply Hamming window (vectorized)
Windowing::applyHammingWindowNEON(A_frame_left, A_frame_right, B_frame_left, B_frame_right, hamming_window, frame_duration);


// NEON-optimized window application
void Windowing::applyHammingWindowNEON(std::vector<double>& A_left, std::vector<double>& A_right,
                                      std::vector<double>& B_left, std::vector<double>& B_right,
                                      const std::vector<double>& hamming_window, size_t frame_duration) {
    size_t j = 0;
    size_t vec_size = frame_duration - (frame_duration % 2); // Process two elements at a time
    
    for (; j < vec_size; j += 2) {
        // Load two window coefficients
        float64x2_t win = vld1q_f64(&hamming_window[j]);

        // Load two samples for Stream A Left
        float64x2_t windowed_A_L = vmulq_f64(samp_A_L, win);
    // Apply window (Stream A Left)
float64x2_t windowed_A_L = vmulq_f64(samp_A_L, win);
vst1q_f64(&A_left[j], windowed_A_L); // Add this line

        // Load two samples for Stream A Right
        float64x2_t windowed_A_R = vmulq_f64(samp_A_R, win);
    // Apply window (Stream A Right)
float64x2_t windowed_A_R = vmulq_f64(samp_A_R, win);
vst1q_f64(&A_right[j], windowed_A_R); // Add this line

        // Load two samples for Stream B Left  
        float64x2_t windowed_B_L = vmulq_f64(samp_B_L, win);
    // Apply window (Stream B Left)
float64x2_t windowed_B_L = vmulq_f64(samp_B_L, win);
vst1q_f64(&B_left[j], windowed_B_L); // Add this line

        // Load two samples for Stream B Right
        float64x2_t windowed_B_R = vmulq_f64(samp_B_R, win);
    // Apply window (Stream B Right)
float64x2_t windowed_B_R = vmulq_f64(samp_B_R, win);
vst1q_f64(&B_right[j], windowed_B_R); // Add this line
    }
    
    // Handle remaining samples
    for (; j < frame_duration; ++j) {
        A_left[j] *= hamming_window[j];
        A_right[j] *= hamming_window[j];
        B_left[j] *= hamming_window[j];
        B_right[j] *= hamming_window[j];
    }
}
c. Leverage Templates and Inline Functions
For repetitive vectorized operations, consider using templates and inline functions to reduce code duplication and enhance performance.

Example: Vectorized Multiplication Function

cpp
Copy code
#include <arm_neon.h>

inline void vector_multiply_add(float* dest, const float* src1, const float* src2, size_t size) {
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t a = vld1q_f32(&src1[i]);
        float32x4_t b = vld1q_f32(&src2[i]);
        float32x4_t result = vmulq_f32(a, b);
        vst1q_f32(&dest[i], result);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        dest[i] = src1[i] * src2[i];
    }
}