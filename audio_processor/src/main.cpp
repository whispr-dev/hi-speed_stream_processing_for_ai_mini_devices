#include "AudioIO.h"
#include "FFTProcessor.h"
#include "FeatureExtractor.h"
#include "Windowing.h"
#include "DeltaProcessor.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <omp.h>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

int main() {
    // File paths
    std::string streamAFile = "../data/streamA_stereo.wav";
    std::string streamBFile = "../data/streamB_stereo.wav";
    std::string deltaFile = "deltaDataStream.bin";
    std::string featureFilePath = "featureMatrix.csv";

    // Audio channels and metadata
    std::vector<double> A_left, A_right, B_left, B_right;
    int fs_A = 0, fs_B = 0;

    // Read and validate WAV files
    AudioIO audioIO;
    if (!audioIO.readStereoWavFileValidated(streamAFile, A_left, A_right, fs_A)) {
        std::cerr << "Error reading Stream A." << std::endl;
        return -1;
    }
    if (!audioIO.readStereoWavFileValidated(streamBFile, B_left, B_right, fs_B)) {
        std::cerr << "Error reading Stream B." << std::endl;
        return -1;
    }

    // Ensure sample rates match
    if (fs_A != fs_B) {
        std::cerr << "Sample rates do not match." << std::endl;
        return -1;
    }

    // Determine processing parameters
    const int frameDuration = 512;
    const int hopSize = frameDuration / 4; // 75% overlap
    const size_t minSize = std::min({A_left.size(), A_right.size(), B_left.size(), B_right.size()});
    const size_t numberOfFrames = (minSize - frameDuration) / hopSize + 1;

    // Resize audio channels to minimum size
    A_left.resize(minSize, 0.0);
    A_right.resize(minSize, 0.0);
    B_left.resize(minSize, 0.0);
    B_right.resize(minSize, 0.0);

    // Initialize DeltaProcessor
    DeltaProcessor deltaProcessor(numberOfFrames);

    struct FrameDeltaSparseMatrices {
        Eigen::SparseMatrix<double> Delta_A_L_B_L;
        Eigen::SparseMatrix<double> Delta_A_L_B_R;
        Eigen::SparseMatrix<double> Delta_A_R_B_L;
        Eigen::SparseMatrix<double> Delta_A_R_B_R;
    };


    // Process audio frames to extract delta matrices
    deltaProcessor.processFrames(A_left, A_right, B_left, B_right, frameDuration, hopSize);

    // Serialize delta data
    deltaProcessor.serializeDeltaData(deltaFile);

    // Initialize FeatureExtractor
    const int Nfft = 512;
    const int numFilters = 26;
    const int numCoefficients = 13;
    const int numBands = 7;
    FeatureExtractor featureExtractor(Nfft, numFilters, numCoefficients, numBands);

    // Extract features from delta data
    Eigen::MatrixXd featureMatrix = Eigen::MatrixXd::Zero(numberOfFrames, numCoefficients);
    #pragma omp parallel for
    for (size_t i = 0; i < deltaProcessor.getDeltaDataStream().size(); ++i) {
        Eigen::VectorXd features = featureExtractor.extractFeatures(deltaProcessor.getDeltaDataStream()[i], numCoefficients, fs_A);
        #pragma omp critical
        featureMatrix.row(i) = features;
    }

    // Standardize feature matrix
    Eigen::MatrixXd standardizedFeatures = FeatureExtractor::standardizeFeatures(featureMatrix);

    // Save features to CSV
    FeatureExtractor::saveFeaturesToCSV(standardizedFeatures, featureFilePath);

    std::cout << "Feature matrix saved to " << featureFilePath << std::endl;

    return 0;
}
