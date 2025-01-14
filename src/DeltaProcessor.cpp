#include "DeltaProcessor.h"
#include "Utils.h"
#include <iostream>


#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif


DeltaProcessor::DeltaProcessor(size_t frames) : number_of_frames(frames) {}

void DeltaProcessor::processFrames(const std::vector<double>& A_left,
                                   const std::vector<double>& A_right,
                                   const std::vector<double>& B_left,
                                   const std::vector<double>& B_right,
                                   int frameDuration, int hopSize) {
    for (size_t i = 0; i < number_of_frames; ++i) {
        if (!Utils::validateFrame(A_left, frameDuration, "Stream A Left", i) ||
            !Utils::validateFrame(A_right, frameDuration, "Stream A Right", i) ||
            !Utils::validateFrame(B_left, frameDuration, "Stream B Left", i) ||
            !Utils::validateFrame(B_right, frameDuration, "Stream B Right", i)) {
            std::cerr << "Invalid frame detected at index: " << i << std::endl;
            continue;
        }

        // Frame processing logic
        FrameDeltaSparseMatrices deltaFrame;
        deltaDataStream.push_back(deltaFrame);
    }
}

void DeltaProcessor::serializeDeltaData(const std::string& deltaFile) const {
    Utils::serializeDeltaDataStream(deltaDataStream, deltaFile, 512);
}
