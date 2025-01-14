#include "FFTProcessor.h"
#include <fftw3.h>
#include <iostream>


// Compute FFTs
std::vector<std::complex<double>> fft_A_L = fftProcessor.computeFFT(A_frame_left);
std::vector<std::complex<double>> fft_A_R = fftProcessor.computeFFT(A_frame_right);
std::vector<std::complex<double>> fft_B_L = fftProcessor.computeFFT(B_frame_left);
std::vector<std::complex<double>> fft_B_R = fftProcessor.computeFFT(B_frame_right);

// Inverse FFT and accumulate
std::vector<double> ifft_A_L = fftProcessor.computeIFFT(fft_A_L);
std::vector<double> ifft_A_R = fftProcessor.computeIFFT(fft_A_R);
std::vector<double> ifft_B_L = fftProcessor.computeIFFT(fft_B_L);
std::vector<double> ifft_B_R = fftProcessor.computeIFFT(fft_B_R);

// Accumulate outputs (vectorized)
Utils::accumulateToOutputNEON(A_left, ifft_A_L, begin_sample, frame_duration);
Utils::accumulateToOutputNEON(A_right, ifft_A_R, begin_sample, frame_duration);
Utils::accumulateToOutputNEON(B_left, ifft_B_L, begin_sample, frame_duration);
Utils::accumulateToOutputNEON(B_right, ifft_B_R, begin_sample, frame_duration);

FFTProcessor::FFTProcessor(int Nfft) : Nfft_(Nfft) {
    // Initialize FFTW plans
}

FFTProcessor::~FFTProcessor() {
    // Destroy FFTW plans and free resources
}

std::vector<std::complex<double>> FFTProcessor::computeFFT(const std::vector<double>& frame) {
    // Implement FFT computation
}

std::vector<double> FFTProcessor::computeIFFT(const std::vector<std::complex<double>>& spectrum) {
    // Implement IFFT computation
}
