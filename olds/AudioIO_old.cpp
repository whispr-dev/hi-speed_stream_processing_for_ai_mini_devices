#include <sndfile.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <arm_neon.h> // Include NEON intrinsics



    // File paths
    std::string streamAFile = "../data/streamA_stereo.wav";
    std::string streamBFile = "../data/streamB_stereo.wav";
    std::string deltaFile = "deltaDataStream.bin";
    std::string featureFilePath = "featureMatrix.csv";
    std::string hashMapFilePath = "frameHashMap.txt";
    
    // Audio channels
    std::vector<double> A_left, A_right, B_left, B_right;
    int fs_A, fs_B;
    
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

    // Validate frames
    if (!Utils::validateFrame(A_frame_left, frame_duration, "Stream A Left", i) ||
        !Utils::validateFrame(A_frame_right, frame_duration, "Stream A Right", i) ||
        !Utils::validateFrame(B_frame_left, frame_duration, "Stream B Left", i) ||
        !Utils::validateFrame(B_frame_right, frame_duration, "Stream B Right", i)) {
        #pragma omp critical
        {
            std::cerr << "Warning: Skipping frame " << i << " due to invalid data." << std::endl;
        }
        continue;
    }

    // Ensure same sample rates
    if (fs_A != fs_B) {
        std::cerr << "Sample rates do not match." << std::endl;
        return -1;
    }
    
    // Determine processing parameters
    int frame_duration = 256;
    int hopSize = frame_duration / 4; // 75% overlap
    size_t min_size = std::min({A_left.size(), A_right.size(), B_left.size(), B_right.size()});
    size_t number_of_frames = (min_size - frame_duration) / hopSize + 1;
    
    // Resize channels to minimum size
    A_left.resize(min_size, 0.0);
    A_right.resize(min_size, 0.0);
    B_left.resize(min_size, 0.0);
    B_right.resize(min_size, 0.0);