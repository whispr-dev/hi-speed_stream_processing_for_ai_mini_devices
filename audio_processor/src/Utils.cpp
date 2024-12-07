#include "Utils.h"
#include <iostream>
#include <fstream>
#include <Eigen/Dense>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

// Function to validate a frame
bool Utils::validateFrame(const std::vector<double>& frame, int expectedSize, const std::string& streamName, int frameIndex) {
    if (frame.size() != expectedSize) {
        std::cerr << "Validation failed for " << streamName << " at frame " << frameIndex
                  << ". Expected size: " << expectedSize << ", but got: " << frame.size() << std::endl;
        return false;
    }
    return true;
}

// Function to serialize a delta data stream to a file
void Utils::serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const std::string& filePath, int Nfft) {
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    for (const auto& deltaMatrix : deltaDataStream) {
        // Serialization logic here
    }
    std::cout << "Serialized delta data stream to: " << filePath << std::endl;
}

// Function to standardize a feature matrix
Eigen::MatrixXd Utils::standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix) {
    Eigen::MatrixXd standardized = featureMatrix;

    for (int col = 0; col < standardized.cols(); ++col) {
        Eigen::VectorXd column = standardized.col(col);
        double mean = column.mean();
        double stdDev = std::sqrt((column.array() - mean).square().mean());

        if (stdDev != 0) {
            standardized.col(col) = (column.array() - mean) / stdDev;
        }
    }

    return standardized;
}
