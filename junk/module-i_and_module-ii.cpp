#include <sndfile.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <arm_neon.h> // Include NEON intrinsics
#include <fstream>
#include <unordered_map>
#include <functional>
#include <omp.h> // For parallel processing

// ------------------------------
// Aligned Allocator for Eigen
// ------------------------------
template <typename T, std::size_t Alignment>
struct aligned_allocator {
    using value_type = T;

    aligned_allocator() noexcept {}
    template <typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }
};

template <typename T, typename U, std::size_t Alignment>
bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) { return true; }

template <typename T, typename U, std::size_t Alignment>
bool operator!=(const aligned_allocator<T, Alignment>& a, const aligned_allocator<U, Alignment>& b) { return !(a == b); }

// ------------------------------
// Data Structures
// ------------------------------

// Define a 4D data stream as a vector of frames, each frame has four sparse matrices
struct FrameSparseMatrices {
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_L_B_L; // Stream A Left convolved with Stream B Left
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_L_B_R; // Stream A Left convolved with Stream B Right
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_R_B_L; // Stream A Right convolved with Stream B Left
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> A_R_B_R; // Stream A Right convolved with Stream B Right

    FrameSparseMatrices(int rows, int cols) 
        : A_L_B_L(rows, cols), A_L_B_R(rows, cols),
          A_R_B_L(rows, cols), A_R_B_R(rows, cols) {}
};

// Define a 4D delta data stream as a vector of frames, each frame has four delta sparse matrices
struct FrameDeltaSparseMatrices {
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Delta_A_L_B_L; // Delta Stream A Left convolved with Stream B Left
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Delta_A_L_B_R; // Delta Stream A Left convolved with Stream B Right
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Delta_A_R_B_L; // Delta Stream A Right convolved with Stream B Left
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> Delta_A_R_B_R; // Delta Stream A Right convolved with Stream B Right

    FrameDeltaSparseMatrices(int rows, int cols) 
        : Delta_A_L_B_L(rows, cols), Delta_A_L_B_R(rows, cols),
          Delta_A_R_B_L(rows, cols), Delta_A_R_B_R(rows, cols) {}
};

// ------------------------------
// Parsing and Preparation Module
// ------------------------------

// Function to serialize a sparse matrix into a string
std::string serializeSparseMatrixToString(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat) {
    std::string serialized;
    serialized.reserve(mat.nonZeros() * 20); // Preallocate memory for efficiency
    
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
            serialized += std::to_string(it.row()) + "," + std::to_string(it.col()) + "," + std::to_string(it.value()) + ";";
        }
    }
    
    return serialized;
}

// Function to serialize FrameDeltaSparseMatrices into a single string
std::string serializeFrameDeltaToString(const FrameDeltaSparseMatrices& frameDelta) {
    std::string serialized;
    serialized += serializeSparseMatrixToString(frameDelta.Delta_A_L_B_L) + "|";
    serialized += serializeSparseMatrixToString(frameDelta.Delta_A_L_B_R) + "|";
    serialized += serializeSparseMatrixToString(frameDelta.Delta_A_R_B_L) + "|";
    serialized += serializeSparseMatrixToString(frameDelta.Delta_A_R_B_R);
    return serialized;
}

// Define a type for the hash map: key is serialized frame string, value is a unique ID
using FrameHashMap = std::unordered_map<std::string, int>;

// Function to build the hash map from deltaDataStream
FrameHashMap buildFrameHashMap(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream) {
    FrameHashMap frameMap;
    int uniqueID = 0;
    
    for (const auto& frameDelta : deltaDataStream) {
        std::string serializedFrame = serializeFrameDeltaToString(frameDelta);
        
        // If the frame is not already in the map, add it with a unique ID
        if (frameMap.find(serializedFrame) == frameMap.end()) {
            frameMap[serializedFrame] = uniqueID++;
        }
    }
    
    std::cout << "Total unique frames: " << uniqueID << std::endl;
    return frameMap;
}

// Function to extract features from FrameDeltaSparseMatrices
Eigen::VectorXd extractFeatures(const FrameDeltaSparseMatrices& frameDelta, int number_of_features) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();
    
    // Example Feature 1: Total non-zero elements across all matrices
    int totalNnz = frameDelta.Delta_A_L_B_L.nonZeros() + 
                   frameDelta.Delta_A_L_B_R.nonZeros() + 
                   frameDelta.Delta_A_R_B_L.nonZeros() + 
                   frameDelta.Delta_A_R_B_R.nonZeros();
    features(0) = static_cast<double>(totalNnz);
    
    // Example Feature 2: Sum of all values across all matrices
    double sumValues = 0.0;
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                sumValues += it.value();
            }
        }
    }
    features(1) = sumValues;
    
    // Example Features 3-12: Additional statistical or domain-specific features
    // For demonstration, we'll populate them with zero. Replace with actual feature calculations.
    for (int i = 2; i < number_of_features; ++i) {
        features(i) = 0.0; // Replace with actual feature calculations
    }
    
    return features;
}

// Function to build feature matrix from deltaDataStream and frameMap
Eigen::MatrixXd buildFeatureMatrix(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const FrameHashMap& frameMap, int number_of_features) {
    Eigen::MatrixXd featureMatrix(frameMap.size(), number_of_features);
    featureMatrix.setZero();
    
    // Iterate through deltaDataStream and accumulate features
    // Parallelizing with OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < deltaDataStream.size(); ++i) {
        std::string serializedFrame = serializeFrameDeltaToString(deltaDataStream[i]);
        auto it = frameMap.find(serializedFrame);
        if (it != frameMap.end()) {
            int id = it->second;
            Eigen::VectorXd features = extractFeatures(deltaDataStream[i], number_of_features);
            
            // Atomic update to prevent race conditions
            #pragma omp critical
            {
                featureMatrix.row(id) += features;
            }
        }
    }
    
    return featureMatrix;
}
