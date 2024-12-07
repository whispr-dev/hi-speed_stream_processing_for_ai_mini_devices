#include "FeatureExtractor.h"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// Constructor
FeatureExtractor::FeatureExtractor(int Nfft, int num_filters, int num_coefficients, int num_bands)
    : Nfft_(Nfft), num_filters_(num_filters), num_coefficients_(num_coefficients), num_bands_(num_bands) {
    precompute_dct_basis();
    precompute_mel_filters();
}

// Precompute DCT basis
void FeatureExtractor::precompute_dct_basis() {
    dct_basis_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            dct_basis_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
        }
    }
}

// Precompute Mel filters
void FeatureExtractor::precompute_mel_filters() {
    mel_filters_.resize(num_filters_ * (Nfft_ / 2 + 1));
    // Fill in filter coefficients (linear or triangular filters in Mel scale)
}

// Extract Features Function
Eigen::VectorXd FeatureExtractor::extractFeatures(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();

    // Implement full feature extraction logic based on delta matrices
    // Placeholder for real computation (FFT, Mel filters, etc.)
    features[0] = 1.0;  // Example feature
    return features;
}
