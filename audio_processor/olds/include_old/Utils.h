#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <Eigen/Dense> // Required for matrix-related operations
#include "DeltaProcessor.h" // Include if FrameDeltaSparseMatrices is used

class Utils {
public:
    // Validate a frame
    static bool validateFrame(const std::vector<double>& frame, int expectedSize, const std::string& streamName, int frameIndex);

    // Serialize a delta data stream to a binary file
    static void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream,
                                         const std::string& filePath, int Nfft);

    // Standardize a feature matrix (column-wise)
    static Eigen::MatrixXd standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix);

    // Save a feature matrix to a CSV file
    static void saveFeatureMatrixToCSV(const Eigen::MatrixXd& matrix, const std::string& filePath,
                                       const std::vector<std::string>& featureNames);
};

#endif // UTILS_H
