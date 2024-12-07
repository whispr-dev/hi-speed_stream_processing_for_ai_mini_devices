#include <cmath>
#include <numeric>
#include <vector>

// Helper function to calculate skewness
double calculateSkewness(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double skewness = 0.0;
    for (const auto& val : data) {
        skewness += std::pow((val - mean) / stddev, 3);
    }
    return skewness / data.size();
}

// Helper function to calculate kurtosis
double calculateKurtosis(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double kurtosis = 0.0;
    for (const auto& val : data) {
        kurtosis += std::pow((val - mean) / stddev, 4);
    }
    return kurtosis / data.size() - 3.0; // Excess kurtosis
}

// Enhanced extractFeatures function
Eigen::VectorXd extractFeaturesEnhanced(const FrameDeltaSparseMatrices& frameDelta, int number_of_features) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();

    // Frequency domain features
    double spectralCentroid = 0.0;
    double spectralBandwidth = 0.0;
    double spectralRollOff = 0.0;

    // Temporal features
    double zeroCrossingRate = 0.0;
    double energy = 0.0;

    // Statistical features
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;

    // Domain-specific features
    double leftRightRatio = 0.0;

    // Iterate over matrices in the frame
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        std::vector<double> values;
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                values.push_back(it.value());
                energy += it.value() * it.value(); // Sum of squared values for energy
            }
        }

        // Frequency domain features (assuming FFT magnitude is stored in `values`)
        double totalMagnitude = std::accumulate(values.begin(), values.end(), 0.0);
        if (totalMagnitude > 0.0) {
            for (size_t i = 0; i < values.size(); ++i) {
                spectralCentroid += i * values[i] / totalMagnitude;
            }

            for (size_t i = 0; i < values.size(); ++i) {
                spectralBandwidth += std::pow(i - spectralCentroid, 2) * values[i] / totalMagnitude;
            }
            spectralBandwidth = std::sqrt(spectralBandwidth);

            double threshold = 0.85 * totalMagnitude;
            double cumulativeSum = 0.0;
            for (size_t i = 0; i < values.size(); ++i) {
                cumulativeSum += values[i];
                if (cumulativeSum >= threshold) {
                    spectralRollOff = i;
                    break;
                }
            }
        }

        // Temporal features
        for (size_t i = 1; i < values.size(); ++i) {
            if ((values[i - 1] > 0.0 && values[i] < 0.0) || (values[i - 1] < 0.0 && values[i] > 0.0)) {
                zeroCrossingRate += 1.0;
            }
        }

        // Statistical features
        mean = totalMagnitude / values.size();
        for (const auto& val : values) {
            variance += std::pow(val - mean, 2);
        }
        variance /= values.size();
        double stddev = std::sqrt(variance);
        skewness = calculateSkewness(values, mean, stddev);
        kurtosis = calculateKurtosis(values, mean, stddev);

        // Domain-specific features
        leftRightRatio += std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    // Normalize domain-specific features
    leftRightRatio /= 4.0; // Assuming 4 matrices (L-B, L-R, R-B, R-R)

    // Assign calculated features to the feature vector
    features(0) = spectralCentroid;
    features(1) = spectralBandwidth;
    features(2) = spectralRollOff;
    features(3) = zeroCrossingRate;
    features(4) = energy;
    features(5) = mean;
    features(6) = variance;
    features(7) = skewness;
    features(8) = kurtosis;
    features(9) = leftRightRatio;

    // Fill remaining feature slots (if any) with zeros or additional features
    for (int i = 10; i < number_of_features; ++i) {
        features(i) = 0.0; // Replace with actual calculations if available
    }

    return features;
}
