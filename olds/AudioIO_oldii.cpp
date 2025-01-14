#include <sndfile.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "AudioIO.h"


class AudioIO {

// Define static file paths
const std::string AudioIO::streamAFile = "../data/streamA_stereo.wav";
const std::string AudioIO::streamBFile = "../data/streamB_stereo.wav";
const std::string AudioIO::deltaFile = "deltaDataStream.bin";
const std::string AudioIO::featureFilePath = "featureMatrix.csv";
const std::string AudioIO::hashMapFilePath = "frameHashMap.txt";


public:
    // Function to read and validate stereo WAV files
    bool readStereoWavFileValidated(const std::string& filePath, std::vector<double>& leftChannel,
                                    std::vector<double>& rightChannel, int& sampleRate) {
        SF_INFO sfinfo = {};
        SNDFILE* infile = sf_open(filePath.c_str(), SFM_READ, &sfinfo);

        if (!infile) {
            std::cerr << "Error: Could not open file " << filePath << std::endl;
            return false;
        }

        if (sfinfo.channels != 2) {
            std::cerr << "Error: File " << filePath << " is not stereo." << std::endl;
            sf_close(infile);
            return false;
        }

        sampleRate = sfinfo.samplerate;
        size_t totalFrames = sfinfo.frames;
        std::vector<double> tempBuffer(totalFrames * 2); // Stereo buffer

        // Read samples into the temporary buffer
        if (sf_readf_double(infile, tempBuffer.data(), totalFrames) != totalFrames) {
            std::cerr << "Error: Failed to read samples from " << filePath << std::endl;
            sf_close(infile);
            return false;
        }
        sf_close(infile);

        // Split stereo buffer into left and right channels
        leftChannel.resize(totalFrames);
        rightChannel.resize(totalFrames);
        for (size_t i = 0; i < totalFrames; ++i) {
            leftChannel[i] = tempBuffer[2 * i];
            rightChannel[i] = tempBuffer[2 * i + 1];
        }

        return true;
    }
};

// Main processing logic
int main() {
    // File paths
    std::string streamAFile = "../data/streamA_stereo.wav";
    std::string streamBFile = "../data/streamB_stereo.wav";

    // Audio channels
    std::vector<double> A_left, A_right, B_left, B_right;
    int fs_A, fs_B;

    // Create an instance of AudioIO
    AudioIO audioIO;

    // Read and validate WAV files
    if (!audioIO.readStereoWavFileValidated(streamAFile, A_left, A_right, fs_A)) {
        std::cerr << "Error reading Stream A." << std::endl;
        return -1;
    }
    if (!audioIO.readStereoWavFileValidated(streamBFile, B_left, B_right, fs_B)) {
        std::cerr << "Error reading Stream B." << std::endl;
        return -1;
    }

    // Ensure the same sample rates
    if (fs_A != fs_B) {
        std::cerr << "Sample rates do not match." << std::endl;
        return -1;
    }

    // Determine processing parameters
    int frame_duration = 256;
    int hopSize = frame_duration / 4; // 75% overlap
    size_t min_size = std::min({A_left.size(), A_right.size(), B_left.size(), B_right.size()});
    size_t number_of_frames = (min_size - frame_duration) / hopSize + 1;

    // Resize channels to the minimum size
    A_left.resize(min_size, 0.0);
    A_right.resize(min_size, 0.0);
    B_left.resize(min_size, 0.0);
    B_right.resize(min_size, 0.0);

    std::cout << "Processing " << number_of_frames << " frames." << std::endl;

    // Further processing logic would go here...

    return 0;
}
