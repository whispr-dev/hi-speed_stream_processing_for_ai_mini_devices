#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "DeltaProcessor.h" // Include if FrameDeltaSparseMatrices is used

class Utils {
public:
    static bool validateFrame(const std::vector<double>& frame, int expectedSize, const std::string& streamName, int frameIndex);
    static void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream,
                                         const std::string& filePath, int Nfft);
    static Eigen::MatrixXd standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix);
    static void saveFeatureMatrixToCSV(const Eigen::MatrixXd& matrix, const std::string& filePath,
                                       const std::vector<std::string>& featureNames);
};

#endif // UTILS_H
