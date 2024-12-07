#ifndef DELTAPROCESSOR_H
#define DELTAPROCESSOR_H

#include "FeatureExtractor.h" // For feature extraction dependencies
#include "Utils.h" // For utilities like serialization
#include <Eigen/Dense> // For matrix operations
#include <vector>
#include <string>

// Forward declarations to minimize includes
class FrameDeltaSparseMatrices;

class DeltaProcessor {
public:
    // Constructor
    DeltaProcessor(size_t numberOfFrames);

    // Methods
    void processFrames(const std::vector<double>& A_left,
                       const std::vector<double>& A_right,
                       const std::vector<double>& B_left,
                       const std::vector<double>& B_right,
                       int frameDuration, int hopSize);

    void serializeDeltaData(const std::string& deltaFile) const;

    const std::vector<FrameDeltaSparseMatrices>& getDeltaDataStream() const;

// In DeltaProcessor.h
private:
    size_t number_of_frames;

private:
    // Members
    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
};

#endif // DELTAPROCESSOR_H
