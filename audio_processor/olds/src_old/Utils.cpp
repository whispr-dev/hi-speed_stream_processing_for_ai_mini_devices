#include "Utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense> // If Eigen-based operations are needed

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
        // Assuming FrameDeltaSparseMatrices can be serialized directly. If not, implement serialization logic.
        outFile.write(reinterpret_cast<const char*>(&deltaMatrix), sizeof(deltaMatrix));
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

// Function to save feature matrix to a CSV file
void Utils::saveFeatureMatrixToCSV(const Eigen::MatrixXd& matrix, const std::string& filePath, const std::vector<std::string>& featureNames) {
    std::ofstream outFile(filePath);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filePath << std::endl;
        return;
    }

    // Write the header row
    for (size_t i = 0; i < featureNames.size(); ++i) {
        outFile << featureNames[i];
        if (i < featureNames.size() - 1) {
            outFile << ",";
        }
    }
    outFile << "\n";

    // Write the matrix rows
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            outFile << matrix(i, j);
            if (j < matrix.cols() - 1) {
                outFile << ",";
            }
        }
        outFile << "\n";
    }

    std::cout << "Saved feature matrix to: " << filePath << std::endl;
}
