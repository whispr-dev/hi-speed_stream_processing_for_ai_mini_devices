#include "AudioIO.h"
#include "FFTProcessor.h"
#include "FeatureExtractor.h"
#include "Windowing.h"
#include "Utils.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <omp.h>

#include "DeltaProcessor.h"
#include "AudioIO.h"


int main() {
    // File paths
    std::string streamAFile = "../data/streamA_stereo.wav";
    std::string streamBFile = "../data/streamB_stereo.wav";
    std::string deltaFile = "deltaDataStream.bin";
    std::string featureFilePath = "featureMatrix.csv";
    std::string hashMapFilePath = "frameHashMap.txt";


    // Example inputs

    // Audio channels
    std::vector<double> A_left, A_right, B_left, B_right;
    int fs_A, fs_B;
    size_t number_of_frames = 1000;
    int frameDuration = 512;
    int hopSize = 256;
    std::string deltaFile = "delta_output.dat";

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

    // Resize channels to minimum size
    A_left.resize(min_size, 0.0);
    A_right.resize(min_size, 0.0);
    B_left.resize(min_size, 0.0);
    B_right.resize(min_size, 0.0);

    // Initialize DeltaProcessor
    DeltaProcessor deltaProcessor(number_of_frames);

    // Process frames
    deltaProcessor.processFrames(A_left, A_right, B_left, B_right, frameDuration, hopSize);

    // Serialize delta data
    deltaProcessor.serializeDeltaData(deltaFile);

    return 0;
}


    // Initialize FFTProcessor and FeatureExtractor
    int Nfft = 512;
    FFTProcessor fftProcessor(Nfft);
    int num_filters = 26; // Example
    int num_coefficients = 13;
    int num_bands = 7;
    FeatureExtractor featureExtractor(Nfft, num_filters, num_coefficients, num_bands);
    
    // ... [Further processing like ML model training] .../


   // Initialize data streams
    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
    deltaDataStream.reserve(number_of_frames);
    
    // Processing Loop
    #pragma omp parallel
    {
        // Thread-local storage
        std::vector<FrameDeltaSparseMatrices> localDeltaData;
        
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < number_of_frames; ++i) {
            int begin_sample = i * hopSize;
            int end_sample = begin_sample + frame_duration;
            
            // Extract frames
            std::vector<double> A_frame_left(A_left.begin() + begin_sample, A_left.begin() + end_sample);
            std::vector<double> A_frame_right(A_right.begin() + begin_sample, A_right.begin() + end_sample);
            std::vector<double> B_frame_left(B_left.begin() + begin_sample, B_left.begin() + end_sample);
            std::vector<double> B_frame_right(B_right.begin() + begin_sample, B_right.begin() + end_sample);
            
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

            
            // Create current frame's delta matrices
            FrameDeltaSparseMatrices deltaFrame;
            // Populate deltaFrame by comparing with previous frame (implement accordingly)
            // ...
            
            // Store in thread-local storage
            localDeltaData.push_back(deltaFrame);
        }
        
        // Merge thread-local delta data into global stream
        #pragma omp critical
        {
            deltaDataStream.insert(deltaDataStream.end(), localDeltaData.begin(), localDeltaData.end());
        }
    }
    
    // Serialize delta data stream
    Utils::serializeDeltaDataStream(deltaDataStream, deltaFile, Nfft);
    std::cout << "Delta data stream serialized to " << deltaFile << std::endl;
    
    // Standardize Feature Matrix
    Eigen::MatrixXd standardizedFeatures = Utils::standardizeFeatureMatrix(featureMatrix);
    
    // Save to CSV
    Utils::saveFeatureMatrixToCSV(standardizedFeatures, featureFilePath, Utils::FEATURE_NAMES);
    std::cout << "Feature matrix saved to " << featureFilePath << std::endl;
    
    // ... [Further processing like ML model training] ...
    


    return 0;
}
