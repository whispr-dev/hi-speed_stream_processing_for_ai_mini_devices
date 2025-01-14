#include "AudioIO.h"
#include <sndfile.h>
#include <iostream>
#include <string>



#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif


// Define static members
const std::string AudioIO::streamAFile = "../data/streamA_stereo.wav";
const std::string AudioIO::streamBFile = "../data/streamB_stereo.wav";

// Constructor implementation
AudioIO::AudioIO() {
    // Initialize member variables (if needed)
}

// Method implementations
bool AudioIO::readStereoWavFileValidated(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate) {
    std::cout << "Reading WAV file: " << filename << std::endl;

    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error reading WAV file: " << sf_strerror(file) << std::endl;
        return false;
    }

    sampleRate = sfInfo.samplerate;
    size_t numFrames = sfInfo.frames;
    left.resize(numFrames);
    right.resize(numFrames);

    std::vector<float> buffer(numFrames * sfInfo.channels);
    sf_readf_float(file, buffer.data(), numFrames);

    for (size_t i = 0; i < numFrames; ++i) {
        left[i] = buffer[i * 2];
        right[i] = buffer[i * 2 + 1];
    }

    sf_close(file);
    return true;
}
