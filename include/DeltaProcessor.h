#ifndef DELTAPROCESSOR_H
#define DELTAPROCESSOR_H

#include "FeatureExtractor.h"
#include "Utils.h"
#include <vector>
#include <string>

// Forward declaration to avoid circular dependencies
class FrameDeltaSparseMatrices;

class DeltaProcessor {
public:
    DeltaProcessor(size_t numberOfFrames);

    void processFrames(const std::vector<double>& A_left,
                       const std::vector<double>& A_right,
                       const std::vector<double>& B_left,
                       const std::vector<double>& B_right,
                       int frameDuration, int hopSize);

    void serializeDeltaData(const std::string& deltaFile) const;
    const std::vector<FrameDeltaSparseMatrices>& getDeltaDataStream() const;

private:
    size_t number_of_frames;
    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
};

#endif // DELTAPROCESSOR_H
