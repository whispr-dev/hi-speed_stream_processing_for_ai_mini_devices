#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <Eigen/Dense>
#include <vector>
#include "DataStructures.h"

class FeatureExtractor {
public:
    FeatureExtractor(int Nfft, int num_filters, int num_coefficients, int num_bands);
    Eigen::VectorXd extractFeatures(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate);

private:
    int Nfft_;
    int num_filters_;
    int num_coefficients_;
    int num_bands_;
    std::vector<float> dct_basis_;
    std::vector<float> mel_filters_;
    std::vector<int> chroma_bins_; // Adjusted to be a vector for flexibility

    void precompute_dct_basis();
    void precompute_mel_filters();
    void map_freq_to_chroma();

    // NEON-optimized internal functions
    void compute_log_mel_energies_neon(float* mel_energies, size_t size);
    void compute_dct_neon(const float* log_mel_energies, const std::vector<float>& dct_basis, float* mfccs);
    void compute_chroma_neon(const float* spectrum, const std::vector<int>& chroma_bins, float* chroma);
    void compute_spectral_contrast_neon(const float* spectrum, float* spectral_contrast);
    void compute_tonnetz_neon(const float* chroma, float* tonnetz);
};

#endif // FEATUREEXTRACTOR_H
