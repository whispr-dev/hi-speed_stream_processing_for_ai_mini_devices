#include "FFTProcessor.h"
#include <fftw3.h>
#include <iostream>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// Constructor
FFTProcessor::FFTProcessor(int Nfft) : Nfft_(Nfft) {
    // Initialize FFTW plans
}

// Destructor
FFTProcessor::~FFTProcessor() {
    // Destroy FFTW plans
}

// Compute FFT
std::vector<std::complex<double>> FFTProcessor::computeFFT(const std::vector<double>& frame) {
    std::vector<std::complex<double>> spectrum(Nfft_);
    // Implement FFT using FFTW
    return spectrum;
}

// Compute IFFT
std::vector<double> FFTProcessor::computeIFFT(const std::vector<std::complex<double>>& spectrum) {
    std::vector<double> frame(Nfft_);
    // Implement IFFT using FFTW
    return frame;
}
