#include "FeatureExtractor.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>


// Constructor
FeatureExtractor::FeatureExtractor(int Nfft, int num_filters, int num_coefficients, int num_bands)
    : Nfft_(Nfft), num_filters_(num_filters), num_coefficients_(num_coefficients), num_bands_(num_bands) {
    precompute_dct_basis();
    precompute_mel_filters();
    map_freq_to_chroma();
}

    // Feature Extraction
    Eigen::MatrixXd featureMatrix = Eigen::MatrixXd::Zero(deltaDataStream.size(), 48); // Example size
    #pragma omp parallel for
    for (size_t i = 0; i < deltaDataStream.size(); ++i) {
        Eigen::VectorXd features = featureExtractor.extractFeatures(deltaDataStream[i], 48, fs_A);
        #pragma omp critical
        {
            featureMatrix.row(i) = features;
            }
        }
    }

// Standardize Feature Matrix
Eigen::MatrixXd standardizedFeatures = Utils::standardizeFeatureMatrix(featureMatrix);

// Save to CSV
 Utils::saveFeatureMatrixToCSV(standardizedFeatures, featureFilePath, Utils::FEATURE_NAMES);
 std::cout << "Feature matrix saved to " << featureFilePath << std::endl;

// Precompute DCT basis
void FeatureExtractor::precompute_dct_basis() {
    // Implement precomputation of DCT basis vectors

}
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
    // Implement precomputation of Mel filter bank
}

void FeatureExtractor::precompute_dct_basis() {
    mel_filters_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            mel_filters_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
            }
        }
    }

// Map frequency bins to Chroma classes
void FeatureExtractor::map_freq_to_chroma() {
    // Implement mapping of frequency bins to chroma classes
}

void FeatureExtractor::precompute_dct_basis() {
    chroma_classes_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            chroma_classes_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
            }
        }
    }

// NEON-optimized logarithm function
float32x4_t FeatureExtractor::compute_logf_approx(float32x4_t x) {
    // Implement or call NEON-optimized log approximation
}

void FeatureExtractor::precompute_dct_basis() {
    log_fnctn_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            log_fnctn_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
            }
        }
    }

// NEON-optimized log Mel energies
void FeatureExtractor::compute_log_mel_energies_neon(float* mel_energies, size_t size) {
    // Implement NEON-optimized log computation
}

void FeatureExtractor::precompute_dct_basis() {
    log_mel_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            log_mel_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
            }
        }
    }

// NEON-optimized DCT
void FeatureExtractor::compute_dct_neon(const float* log_mel_energies, const std::vector<float>& dct_basis, float* mfccs) {
    // Implement NEON-optimized DCT
}

void FeatureExtractor::precompute_dct_basis() {
    dct_neon_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            dct_neon_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
        }
    }
}

// NEON-optimized Chroma computation
void FeatureExtractor::compute_chroma_neon(const float* spectrum, const int* chroma_bins, float* chroma) {
    // Implement NEON-optimized Chroma features
}

void FeatureExtractor::precompute_dct_basis() {
    chroma_neon_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            chroma_neon_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
        }
    }
}

// NEON-optimized Spectral Contrast
void FeatureExtractor::compute_spectral_contrast_neon(const float* spectrum, float* spectral_contrast) {
    // Implement NEON-optimized Spectral Contrast
}

void FeatureExtractor::precompute_dct_basis() {
    sc_neon_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            sc_neon_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
        }
    }
}

// NEON-optimized Tonnetz computation
void FeatureExtractor::compute_tonnetz_neon(const float* chroma, float* tonnetz) {
    // Implement NEON-optimized Tonnetz
}

void FeatureExtractor::precompute_dct_basis() {
    tometz_neon_.resize(num_coefficients_ * num_filters_);
    for (int i = 0; i < num_coefficients_; ++i) {
        for (int j = 0; j < num_filters_; ++j) {
            tometz_neon_[i * num_filters_ + j] =
                std::cos(M_PI * i / num_filters_ * (j + 0.5));
        }
    }
}

// Extract Features Function
Eigen::VectorXd FeatureExtractor::extractFeatures(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();
    
    // Aggregate spectrum
    std::vector<std::complex<double>> aggregatedSpectrum;
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                aggregatedSpectrum.emplace_back(it.value(), 0.0);
            }
        }
    }
    
        // Convert to float array
        std::vector<float> spectrum_float(aggregatedSpectrum.size());
        for (size_t i = 0; i < aggregatedSpectrum.size(); ++i) {
            spectrum_float[i] = static_cast<float>(std::abs(aggregatedSpectrum[i]));
            }
        }
    }

    // Compute FFT (assuming FFTProcessor is another module)
    // FFTProcessor fftProcessor(Nfft_);
    // std::vector<std::complex<double>> spectrum = fftProcessor.computeFFT(spectrum_float);
    
    // Placeholder for spectrum
    std::vector<std::complex<double>> spectrum; // Fill with actual FFT results
    
    // Compute power spectrum
    std::vector<float> powerSpectrum(Nfft_ / 2 + 1, 0.0f);
    for (int k = 0; k < Nfft_ / 2 + 1; ++k) {
        powerSpectrum[k] = static_cast<float>(std::pow(std::abs(spectrum[k]), 2));
    }

    // Apply Mel filter banks (vectorized)
    std::vector<float> mel_energies(num_filters_, 0.0f);
    for (int m = 0; m < num_filters_; ++m) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 3 < Nfft_ / 2 + 1; k += 4) {
            float32x4_t mel = vld1q_f32(&mel_filters_[m * (Nfft_ / 2 + 1) + k]);
            float32x4_t spec = vld1q_f32(&powerSpectrum[k]);
            sum = vmlaq_f32(sum, spec, mel);
        }
        // Horizontal sum
        float32x2_t sum_pair = vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        float32x1_t final_sum = vpadd_f32(sum_pair, sum_pair);
        mel_energies[m] = vget_lane_f32(final_sum, 0);
        
        // Handle remaining bins
        for (; k < Nfft_ / 2 + 1; ++k) {
            mel_energies[m] += powerSpectrum[k] * mel_filters_[m * (Nfft_ / 2 + 1) + k];
        }
    }
    
    // Compute log Mel energies (vectorized)
    compute_log_mel_energies_neon(mel_energies.data(), mel_energies.size());
    
    // Compute MFCCs using vectorized DCT
    std::vector<float> mfccs(num_coefficients_, 0.0f);
    compute_dct_neon(mel_energies.data(), dct_basis_, mfccs.data(), num_filters_, num_coefficients_);
    
    // Compute Chroma Features (vectorized)
    std::vector<float> chroma(12, 0.0f);
    compute_chroma_neon(spectrum_float.data(), chroma_bins_, chroma.data());
    
    // Compute Spectral Contrast (vectorized)
    std::vector<float> spectral_contrast(num_bands_, 0.0f);
    compute_spectral_contrast_neon(spectrum_float.data(), spectral_contrast.data());
    
    // Compute Tonnetz Features (vectorized)
    std::vector<float> tonnetz(6, 0.0f);
    compute_tonnetz_neon(chroma.data(), tonnetz.data());
    
    // Assign existing features (spectral centroid, etc.) - vectorized implementations needed
    // Placeholder assignments
    features(0) = 0.0; // spectral_centroid
    features(1) = 0.0; // spectral_bandwidth
    features(2) = 0.0; // spectral_rolloff
    features(3) = 0.0; // zero_crossing_rate
    features(4) = 0.0; // energy
    features(5) = 0.0; // mean
    features(6) = 0.0; // variance
    features(7) = 0.0; // skewness
    features(8) = 0.0; // kurtosis
    features(9) = 0.0; // left_right_ratio
    
    // Assign vectorized features
    int featureIdx = 10;
    for (const auto& mfcc : mfccs) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = mfcc;
        }
    }
    for (const auto& ch : chroma) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = ch;
        }
    }
    for (const auto& sc : spectral_contrast) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = sc;
        }
    }
    for (const auto& tn : tonnetz) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = tn;
        }
    }
    
    // Fill remaining feature slots with zeros or additional features
    for (; featureIdx < number_of_features; ++featureIdx) {
        features(featureIdx) = 0.0; // Replace with actual calculations if available
    }
    
    return features;
}
