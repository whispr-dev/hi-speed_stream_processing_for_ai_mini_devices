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
#include <cassert>

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
// Validation Functions
// ------------------------------

// Function to validate WAV file integrity and properties
bool validateWavFile(const SF_INFO& sfInfo, const std::string& filename) {
    bool isValid = true;

    // Check if the file was successfully opened
    if (sfInfo.frames <= 0) {
        std::cerr << "Error: WAV file " << filename << " contains no frames." << std::endl;
        isValid = false;
    }

    // Check if the file has exactly two channels (stereo)
    if (sfInfo.channels != 2) {
        std::cerr << "Error: WAV file " << filename << " is not stereo. Channels found: " << sfInfo.channels << std::endl;
        isValid = false;
    }

    // Check for supported formats (e.g., only WAV)
    if ((sfInfo.format & SF_FORMAT_TYPEMASK) != SF_FORMAT_WAV) {
        std::cerr << "Error: WAV file " << filename << " is not in WAV format." << std::endl;
        isValid = false;
    }

    // Optionally, check for specific subformats (e.g., PCM)
    int subFormat = sfInfo.format & SF_FORMAT_SUBMASK;
    if (subFormat != SF_FORMAT_PCM_16 &&
        subFormat != SF_FORMAT_PCM_24 &&
        subFormat != SF_FORMAT_PCM_32 &&
        subFormat != SF_FORMAT_FLOAT) {
        std::cerr << "Warning: WAV file " << filename << " is not in a standard PCM format. Proceeding with caution." << std::endl;
    }

    return isValid;
}

// Function to validate frame data
bool validateFrame(const std::vector<double>& frame, int expectedSize, const std::string& streamName, int frameIndex) {
    if (frame.size() != expectedSize) {
        std::cerr << "Error: Frame " << frameIndex << " in " << streamName << " has incorrect size. Expected: " << expectedSize << ", Got: " << frame.size() << std::endl;
        return false;
    }

    // Check for NaNs or infinities
    for (size_t i = 0; i < frame.size(); ++i) {
        if (std::isnan(frame[i]) || std::isinf(frame[i])) {
            std::cerr << "Error: Frame " << frameIndex << " in " << streamName << " contains invalid samples at index " << i << "." << std::endl;
            return false;
        }
    }

    return true;
}

// Function to validate FFT results
bool validateFFTResult(const std::vector<std::complex<double>>& spectrum, const std::string& operation, int frameIndex) {
    for (size_t i = 0; i < spectrum.size(); ++i) {
        if (std::isnan(spectrum[i].real()) || std::isnan(spectrum[i].imag()) ||
            std::isinf(spectrum[i].real()) || std::isinf(spectrum[i].imag())) {
            std::cerr << "Error: " << operation << " resulted in invalid spectrum at frame " << frameIndex << ", index " << i << "." << std::endl;
            return false;
        }
    }
    return true;
}

// Function to validate IFFT results
bool validateIFFTResult(const std::vector<double>& timeDomain, const std::string& operation, int frameIndex) {
    for (size_t i = 0; i < timeDomain.size(); ++i) {
        if (std::isnan(timeDomain[i]) || std::isinf(timeDomain[i])) {
            std::cerr << "Error: " << operation << " resulted in invalid time-domain signal at frame " << frameIndex << ", sample " << i << "." << std::endl;
            return false;
        }
    }
    return true;
}

// Function to validate CSV file writing
bool validateCSVExport(std::ofstream& ofs, const std::string& filename) {
    if (!ofs.good()) {
        std::cerr << "Error: Failed to write to CSV file " << filename << "." << std::endl;
        return false;
    }
    return true;
}

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

// Function to calculate skewness
double calculateSkewness(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double skewness = 0.0;
    for (const auto& val : data) {
        skewness += std::pow((val - mean) / stddev, 3);
    }
    return skewness / data.size();
}

// Function to calculate kurtosis
double calculateKurtosis(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double kurtosis = 0.0;
    for (const auto& val : data) {
        kurtosis += std::pow((val - mean) / stddev, 4);
    }
    return kurtosis / data.size() - 3.0; // Excess kurtosis
}

// Function to compute delta between two sparse matrices
Eigen::SparseMatrix<double, Eigen::RowMajor, int> computeDelta(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& current, 
                                                               const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& previous, 
                                                               double threshold) {
    // Compute the difference
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> delta = current - previous;
    
    // Apply thresholding to ignore minor changes
    for (int k=0; k < delta.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(delta, k); it; ++it) {
            if (std::abs(it.value()) < threshold) {
                it.valueRef() = 0.0;
            }
        }
    }
    
    // Prune zero entries
    delta.prune(0.0);
    
    return delta;
}

// ------------------------------
// Feature Extraction Functions
// ------------------------------

// Function to create Mel filter banks
std::vector<std::vector<double>> createMelFilterBanks(int numFilters, int Nfft, int sampleRate) {
    std::vector<std::vector<double>> melFilters(numFilters, std::vector<double>(Nfft / 2 + 1, 0.0));

    // Convert frequency to Mel
    auto freqToMel = [](double freq) -> double {
        return 2595.0 * std::log10(1.0 + freq / 700.0);
    };

    // Convert Mel to frequency
    auto melToFreq = [](double mel) -> double {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    };

    double melMin = freqToMel(0.0);
    double melMax = freqToMel(sampleRate / 2.0);
    std::vector<double> melPoints(numFilters + 2);
    double melStep = (melMax - melMin) / (numFilters + 1);
    for (int i = 0; i < numFilters + 2; ++i) {
        melPoints[i] = melMin + i * melStep;
    }

    std::vector<double> freqPoints(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        freqPoints[i] = melToFreq(melPoints[i]);
    }

    std::vector<int> bin(numFilters + 2);
    for (int i = 0; i < numFilters + 2; ++i) {
        bin[i] = static_cast<int>(std::floor((Nfft + 1) * freqPoints[i] / sampleRate));
    }

    for (int m = 1; m <= numFilters; ++m) {
        for (int k = bin[m - 1]; k < bin[m]; ++k) {
            melFilters[m - 1][k] = (k - bin[m - 1]) / static_cast<double>(bin[m] - bin[m - 1]);
        }
        for (int k = bin[m]; k < bin[m + 1]; ++k) {
            melFilters[m - 1][k] = (bin[m + 1] - k) / static_cast<double>(bin[m + 1] - bin[m]);
        }
    }

    return melFilters;
}

// Function to compute MFCCs from power spectrum
std::vector<double> computeMFCCs(const std::vector<std::complex<double>>& spectrum, int numFilters, int numCoefficients, int sampleRate) {
    int Nfft = spectrum.size();
    std::vector<double> powerSpectrum(Nfft / 2 + 1, 0.0);
    for (int k = 0; k < Nfft / 2 + 1; ++k) {
        powerSpectrum[k] = std::pow(std::abs(spectrum[k]), 2);
    }

    std::vector<std::vector<double>> melFilters = createMelFilterBanks(numFilters, Nfft, sampleRate);
    std::vector<double> melEnergies(numFilters, 0.0);
    for (int m = 0; m < numFilters; ++m) {
        for (int k = 0; k < Nfft / 2 + 1; ++k) {
            melEnergies[m] += powerSpectrum[k] * melFilters[m][k];
        }
        melEnergies[m] = melEnergies[m] > 0.0 ? std::log(melEnergies[m]) : 0.0;
    }

    // Compute DCT of log mel energies
    std::vector<double> mfccs(numCoefficients, 0.0);
    for (int c = 0; c < numCoefficients; ++c) {
        for (int m = 0; m < numFilters; ++m) {
            mfccs[c] += melEnergies[m] * std::cos(M_PI * c * (m + 0.5) / numFilters);
        }
    }

    return mfccs;
}

// Function to compute Chroma Features from power spectrum
std::vector<double> computeChroma(const std::vector<std::complex<double>>& spectrum, int sampleRate) {
    int Nfft = spectrum.size();
    std::vector<double> chroma(12, 0.0);
    
    for (int k = 0; k < Nfft; ++k) {
        double frequency = static_cast<double>(k) * sampleRate / Nfft;
        if (frequency < 27.5 || frequency > 4186.0) continue; // Human hearing range (A0 to C8)
        
        // Convert frequency to MIDI note
        double midi = 69 + 12 * std::log2(frequency / 440.0);
        int pitchClass = static_cast<int>(std::round(midi)) % 12;
        chroma[pitchClass] += std::pow(std::abs(spectrum[k]), 2);
    }
    
    return chroma;
}

// Function to compute Spectral Contrast from power spectrum
std::vector<double> computeSpectralContrast(const std::vector<std::complex<double>>& spectrum, int numBands) {
    int Nfft = spectrum.size();
    std::vector<double> spectralContrast(numBands, 0.0);
    
    int bandSize = (Nfft / 2 + 1) / numBands;
    
    for (int b = 0; b < numBands; ++b) {
        double maxVal = 0.0;
        double minVal = std::pow(10, 100); // Initialize to a large value
        for (int k = b * bandSize; k < (b + 1) * bandSize && k < Nfft / 2 + 1; ++k) {
            double magnitude = std::pow(std::abs(spectrum[k]), 2);
            if (magnitude > maxVal) maxVal = magnitude;
            if (magnitude < minVal) minVal = magnitude;
        }
        if (maxVal > 0.0) {
            spectralContrast[b] = 10 * std::log10(maxVal / (minVal + 1e-10));
        } else {
            spectralContrast[b] = 0.0;
        }
    }
    
    return spectralContrast;
}

// Placeholder function for Tonnetz features
std::vector<double> computeTonnetz(const std::vector<double>& chroma) {
    // Implement Tonnetz calculation based on chroma features
    // This typically involves creating a tonal centroid map
    // For demonstration, return a zero vector
    return std::vector<double>(6, 0.0); // Tonnetz typically has 6 dimensions
}

// Enhanced extractFeatures function with MFCCs, Chroma, Spectral Contrast, and Tonnetz
Eigen::VectorXd extractFeaturesEnhanced(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();
    
    // Existing features
    double spectralCentroid = 0.0;
    double spectralBandwidth = 0.0;
    double spectralRollOff = 0.0;
    double zeroCrossingRate = 0.0;
    double energy = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double leftRightRatio = 0.0;
    
    // New Features
    int numMFCC = 13;
    int numChroma = 12;
    int numSpectralContrast = 7;
    int numTonnetz = 6;
    
    // Temporary storage for aggregated data
    std::vector<std::complex<double>> aggregatedSpectrum;
    
    // Iterate over matrices in the frame and aggregate spectra
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                // Reconstruct the complex spectrum (assuming amplitude only)
                aggregatedSpectrum.emplace_back(it.value(), 0.0);
            }
        }
    }
    
    // Compute FFT of the aggregated spectrum
    int Nfft = 512; // Adjust based on your FFT size
    fftw_complex* in_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_complex* out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_plan plan = fftw_plan_dft_1d(Nfft, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int n = 0; n < Nfft; ++n) {
        if (n < aggregatedSpectrum.size()) {
            in_fft[n][0] = aggregatedSpectrum[n].real();
            in_fft[n][1] = aggregatedSpectrum[n].imag();
        } else {
            in_fft[n][0] = 0.0;
            in_fft[n][1] = 0.0;
        }
    }
    
    fftw_execute(plan);
    
    std::vector<std::complex<double>> spectrum(Nfft);
    for (int n = 0; n < Nfft; ++n) {
        spectrum[n] = std::complex<double>(out_fft[n][0], out_fft[n][1]);
    }
    
    fftw_destroy_plan(plan);
    fftw_free(in_fft);
    fftw_free(out_fft);
    
    // Compute Power Spectrum
    std::vector<double> powerSpectrum(Nfft / 2 + 1, 0.0);
    for (int k = 0; k < Nfft / 2 + 1; ++k) {
        powerSpectrum[k] = std::pow(std::abs(spectrum[k]), 2);
    }
    
    // Compute MFCCs
    std::vector<double> mfccs = computeMFCCs(spectrum, 26, numMFCC, sampleRate);
    
    // Compute Chroma Features
    std::vector<double> chroma = computeChroma(spectrum, sampleRate);
    
    // Compute Spectral Contrast
    std::vector<double> spectralContrast = computeSpectralContrast(spectrum, numSpectralContrast);
    
    // Compute Tonnetz Features
    std::vector<double> tonnetz = computeTonnetz(chroma);
    
    // Assign existing features
    // [Existing feature calculations...]
    // Assigning only spectral centroid, bandwidth, etc.
    // Update based on the new aggregated spectrum if needed
    
    // For demonstration, let's assume we have separate calculations
    // Here we simply assign the existing variables (you should calculate them based on aggregatedSpectrum)
    // Replace these with actual calculations as needed
    
    // Assign calculated features to the feature vector
    features(0) = spectralCentroid;
    features(1) = spectralBandwidth;
    features(2) = spectralRollOff;
    features(3) = zeroCrossingRate;
    features(4) = energy;
    features(5) = mean;
    features(6) = variance;
    features(7) = skewness;
    features(8) = kurtosis;
    features(9) = leftRightRatio;
    
    // Assign new features
    int featureIdx = 10;
    for (const auto& mfcc : mfccs) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = mfcc;
        }
    }
    for (const auto& ch : chroma) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = ch;
        }
    }
    for (const auto& sc : spectralContrast) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = sc;
        }
    }
    for (const auto& tn : tonnetz) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = tn;
        }
    }
    
    // Fill remaining feature slots with zeros or additional features
    for (; featureIdx < number_of_features; ++featureIdx) {
        features(featureIdx) = 0.0; // Replace with actual calculations if available
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
            Eigen::VectorXd features = extractFeaturesEnhanced(deltaDataStream[i], number_of_features, 44100); // Assuming sampleRate = 44100; adjust as needed
            
            // Validate features for NaNs or infinities
            bool validFeatures = true;
            for (int f = 0; f < features.size(); ++f) {
                if (std::isnan(features(f)) || std::isinf(features(f))) {
                    #pragma omp critical
                    {
                        std::cerr << "Error: Feature " << f << " in frame " << i << " is invalid (NaN or Inf)." << std::endl;
                    }
                    validFeatures = false;
                    break;
                }
            }

            if (validFeatures) {
                // Atomic update to prevent race conditions
                #pragma omp critical
                {
                    featureMatrix.row(id) += features;
                }
            }
        }
    }

    return featureMatrix;
}

// Function to save the feature matrix to a CSV file with headers
bool saveFeatureMatrixToCSV(const Eigen::MatrixXd& featureMatrix, const std::string& filename, const std::vector<std::string>& featureNames) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Unable to open file for writing feature matrix: " << filename << std::endl;
        return false;
    }

    // Write the header
    for (size_t i = 0; i < featureNames.size(); ++i) {
        ofs << featureNames[i];
        if (i != featureNames.size() - 1)
            ofs << ",";
    }
    ofs << "\n";

    // Write the data
    for (int i = 0; i < featureMatrix.rows(); ++i) {
        for (int j = 0; j < featureMatrix.cols(); ++j) {
            ofs << featureMatrix(i, j);
            if (j != featureMatrix.cols() - 1)
                ofs << ",";
        }
        ofs << "\n";
    }

    // Validate CSV export
    bool exportSuccess = validateCSVExport(ofs, filename);
    ofs.close();
    return exportSuccess;
}

// Function to serialize a sparse matrix
void serializeSparseMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat, std::ofstream& ofs) {
    int rows = mat.rows();
    int cols = mat.cols();
    int nnz = mat.nonZeros();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&nnz), sizeof(int));

    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            double value = it.value();
            ofs.write(reinterpret_cast<const char*>(&row), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&col), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }
}

// Function to deserialize a sparse matrix
Eigen::SparseMatrix<double, Eigen::RowMajor, int> deserializeSparseMatrix(std::ifstream& ifs, int Nfft) {
    int rows, cols, nnz;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&nnz), sizeof(int));

    Eigen::SparseMatrix<double, Eigen::RowMajor, int> mat(rows, cols);
    mat.reserve(nnz);

    for (int i = 0; i < nnz; ++i) {
        int row, col;
        double value;
        ifs.read(reinterpret_cast<char*>(&row), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&col), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&value), sizeof(double));
        mat.insert(row, col) = value;
    }

    mat.makeCompressed();
    return mat;
}

// Function to serialize the entire delta data stream
void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const std::string& filename, int Nfft) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Failed to open file for serialization: " << filename << std::endl;
        return;
    }

    int numFrames = deltaDataStream.size();
    ofs.write(reinterpret_cast<const char*>(&numFrames), sizeof(int));

    for (const auto& frame : deltaDataStream) {
        serializeSparseMatrix(frame.Delta_A_L_B_L, ofs);
        serializeSparseMatrix(frame.Delta_A_L_B_R, ofs);
        serializeSparseMatrix(frame.Delta_A_R_B_L, ofs);
        serializeSparseMatrix(frame.Delta_A_R_B_R, ofs);
    }

    ofs.close();
}

// Function to deserialize the entire delta data stream
std::vector<FrameDeltaSparseMatrices> deserializeDeltaDataStream(const std::string& filename, int Nfft) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Failed to open file for deserialization: " << filename << std::endl;
        return {};
    }

    int numFrames;
    ifs.read(reinterpret_cast<char*>(&numFrames), sizeof(int));

    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
    deltaDataStream.reserve(numFrames);

    for (int i = 0; i < numFrames; ++i) {
        FrameDeltaSparseMatrices deltaFrame(Nfft, Nfft);
        deltaFrame.Delta_A_L_B_L = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_L_B_R = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_R_B_L = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_R_B_R = deserializeSparseMatrix(ifs, Nfft);
        deltaDataStream.push_back(deltaFrame);
    }

    ifs.close();
    return deltaDataStream;
}

// Function to reconstruct data stream from delta data stream
std::vector<FrameSparseMatrices> reconstructDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, int Nfft) {
    std::vector<FrameSparseMatrices> reconstructedDataStream;
    reconstructedDataStream.reserve(deltaDataStream.size());

    // Initialize previous frame matrices for delta computation (start with zero matrices)
    FrameSparseMatrices previousFrame(Nfft, Nfft);
    previousFrame.A_L_B_L.setZero();
    previousFrame.A_L_B_R.setZero();
    previousFrame.A_R_B_L.setZero();
    previousFrame.A_R_B_R.setZero();

    for (const auto& deltaFrame : deltaDataStream) {
        FrameSparseMatrices currentFrame(Nfft, Nfft);

        // Add deltas to previous frame
        currentFrame.A_L_B_L = previousFrame.A_L_B_L + deltaFrame.Delta_A_L_B_L;
        currentFrame.A_L_B_R = previousFrame.A_L_B_R + deltaFrame.Delta_A_L_B_R;
        currentFrame.A_R_B_L = previousFrame.A_R_B_L + deltaFrame.Delta_A_R_B_L;
        currentFrame.A_R_B_R = previousFrame.A_R_B_R + deltaFrame.Delta_A_R_B_R;

        // Store the current frame
        reconstructedDataStream.push_back(currentFrame);

        // Update previous frame
        previousFrame = currentFrame;
    }

    return reconstructedDataStream;
}

// ------------------------------
// Feature Names Definition
// ------------------------------

// Define feature names in the order they are extracted
const std::vector<std::string> FEATURE_NAMES = {
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "zero_crossing_rate",
    "energy",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "left_right_ratio",
    // MFCCs
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5",
    "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10",
    "mfcc_11", "mfcc_12", "mfcc_13",
    // Chroma Features
    "chroma_0", "chroma_1", "chroma_2", "chroma_3",
    "chroma_4", "chroma_5", "chroma_6", "chroma_7",
    "chroma_8", "chroma_9", "chroma_10", "chroma_11",
    // Spectral Contrast
    "spectral_contrast_0", "spectral_contrast_1",
    "spectral_contrast_2", "spectral_contrast_3",
    "spectral_contrast_4", "spectral_contrast_5",
    "spectral_contrast_6",
    // Tonnetz Features
    "tonnetz_0", "tonnetz_1", "tonnetz_2",
    "tonnetz_3", "tonnetz_4", "tonnetz_5"
    // Add more feature names if number_of_features > 48
};

// ------------------------------
// Final Preparation Module
// ------------------------------

// Function to standardize the feature matrix (zero mean, unit variance)
Eigen::MatrixXd standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix) {
    Eigen::MatrixXd standardized = featureMatrix;
    int numFeatures = featureMatrix.cols();

    for (int i = 0; i < numFeatures; ++i) {
        double mean = featureMatrix.col(i).mean();
        double variance = (featureMatrix.col(i).array() - mean).square().sum() / (featureMatrix.rows() - 1);
        double stddev = std::sqrt(variance);

        if (stddev > 0.0) {
            standardized.col(i) = (featureMatrix.col(i).array() - mean) / stddev;
        } else {
            standardized.col(i).setZero(); // If stddev is zero, set the feature to zero
        }
    }

    return standardized;
}

// Function to perform PCA on the standardized feature matrix
Eigen::MatrixXd performPCA(const Eigen::MatrixXd& standardized, int numComponents) {
    // Compute covariance matrix
    Eigen::MatrixXd cov = (standardized.adjoint() * standardized) / double(standardized.rows() - 1);

    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);

    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Error: Eigen decomposition failed during PCA." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the eigenvalues and eigenvectors
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().reverse(); // Descending order
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().rowwise().reverse();

    // Select the top 'numComponents' eigenvectors
    Eigen::MatrixXd projection = eigenVectors.leftCols(numComponents);

    // Project the data onto the principal components
    Eigen::MatrixXd reduced = standardized * projection;

    return reduced;
}

// ------------------------------
// WAV File Operations
// ------------------------------

// Function to read a stereo WAV file (with validation)
bool readStereoWavFileValidated(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate) {
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error opening WAV file: " << filename << std::endl;
        return false;
    }

    // Validate WAV file
    if (!validateWavFile(sfInfo, filename)) {
        sf_close(file);
        return false;
    }

    sampleRate = sfInfo.samplerate;
    int numChannels = sfInfo.channels;
    int numFrames = sfInfo.frames;

    std::vector<double> tempData(numFrames * numChannels);
    sf_count_t framesRead = sf_readf_double(file, tempData.data(), numFrames);
    sf_close(file);

    if (framesRead != numFrames) {
        std::cerr << "Warning: Expected to read " << numFrames << " frames from " << filename << ", but read " << framesRead << " frames." << std::endl;
    }

    if (numChannels != 2) {
        std::cerr << "Error: Input WAV file " << filename << " is not stereo." << std::endl;
        return false;
    }

    left.resize(numFrames);
    right.resize(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        left[i] = tempData[i * numChannels];
        right[i] = tempData[i * numChannels + 1];
    }

    // Additional validation: Check for NaNs or infinities in the audio data
    for (int i = 0; i < numFrames; ++i) {
        if (std::isnan(left[i]) || std::isinf(left[i])) {
            std::cerr << "Error: Stream Left channel contains invalid sample at frame " << i << "." << std::endl;
            return false;
        }
        if (std::isnan(right[i]) || std::isinf(right[i])) {
            std::cerr << "Error: Stream Right channel contains invalid sample at frame " << i << "." << std::endl;
            return false;
        }
    }

    return true;
}

// Function to write a WAV file (mono)
bool writeWavFileValidated(const std::string& filename, const std::vector<double>& data, int sampleRate) {
    SF_INFO sfInfo;
    sfInfo.channels = 1;
    sfInfo.samplerate = sampleRate;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    if (!file) {
        std::cerr << "Error writing WAV file: " << filename << std::endl;
        return false;
    }

    // Convert double data to short for 16-bit PCM
    std::vector<short> intData(data.size());
    size_t i = 0;

    // Use NEON to process two samples at a time
    for (; i + 1 < data.size(); i += 2) {
        double sample1 = data[i];
        double sample2 = data[i + 1];

        // Clipping
        if (sample1 > 1.0) sample1 = 1.0;
        if (sample1 < -1.0) sample1 = -1.0;
        if (sample2 > 1.0) sample2 = 1.0;
        if (sample2 < -1.0) sample2 = -1.0;

        // Convert to int16
        short intSample1 = static_cast<short>(sample1 * 32767);
        short intSample2 = static_cast<short>(sample2 * 32767);

        // Load into NEON register
        int16x4_t samples = vld1q_s16(reinterpret_cast<const short*>(&intSample1));

        // Store using NEON
        vst1q_s16(&intData[i], samples);
    }

    // Handle remaining sample if size is odd
    for (; i < data.size(); ++i) {
        double sample = data[i];
        if (sample > 1.0) sample = 1.0;
        if (sample < -1.0) sample = -1.0;
        intData[i] = static_cast<short>(sample * 32767);
    }

    // Write data to WAV file
    sf_count_t framesWritten = sf_writef_short(file, intData.data(), data.size());
    sf_close(file);

    if (framesWritten != static_cast<sf_count_t>(data.size())) {
        std::cerr << "Warning: Expected to write " << data.size() << " frames to " << filename << ", but wrote " << framesWritten << " frames." << std::endl;
    }

    return true;
}

// ------------------------------
// Processing Functions
// ------------------------------

// Function to compute FFT using FFTW and return spectrum
std::vector<std::complex<double>> computeFFT(const std::vector<double>& frame, fftw_plan& plan, fftw_complex* in, fftw_complex* out) {
    int N = frame.size();
    for (int n = 0; n < N; ++n) {
        in[n][0] = frame[n];
        in[n][1] = 0.0;
    }

    fftw_execute(plan);

    std::vector<std::complex<double>> spectrum(N);
    for (int n = 0; n < N; ++n) {
        spectrum[n] = std::complex<double>(out[n][0], out[n][1]);
    }

    return spectrum;
}

// Function to compute IFFT using FFTW and return time-domain signal
std::vector<double> computeIFFT(const std::vector<std::complex<double>>& spectrum, fftw_plan& plan, fftw_complex* in, double* out) {
    int N = spectrum.size();
    for (int n = 0; n < N; ++n) {
        in[n][0] = spectrum[n].real();
        in[n][1] = spectrum[n].imag();
    }

    fftw_execute(plan);

    std::vector<double> timeDomain(N);
    for (int n = 0; n < N; ++n) {
        timeDomain[n] = out[n] / N; // Normalize IFFT
    }

    return timeDomain;
}

// Linear interpolation function
std::vector<double> linearInterpolate(const std::vector<double>& input, int targetSize) {
    int inputSize = input.size();
    std::vector<double> output(targetSize, 0.0);
    for (int i = 0; i < targetSize; ++i) {
        double position = static_cast<double>(i) * (inputSize - 1) / (targetSize - 1);
        int index = static_cast<int>(floor(position));
        double frac = position - index;
        if (index + 1 < inputSize) {
            output[i] = input[index] * (1.0 - frac) + input[index + 1] * frac;
        } else {
            output[i] = input[index];
        }
    }
    return output;
}

// Optimized function to flatten carrier spectrum using NEON
std::vector<std::complex<double>> flattenCarrierSpectrumNEON(const std::vector<std::complex<double>>& carrierSpectrum, const std::vector<double>& envelope) {
    size_t size = carrierSpectrum.size();
    std::vector<std::complex<double>> flattenedSpectrum(size);
    size_t i = 0;
    
    for (; i + 1 < size; i += 2) {
        // Load two envelope values
        float64x2_t env = vld1q_f64(&envelope[i]);

        // Load two carrier spectrum real parts
        float64x2_t real_part = vld1q_f64(reinterpret_cast<const double*>(&carrierSpectrum[i].real()));

        // Load two carrier spectrum imaginary parts
        float64x2_t imag_part = vld1q_f64(reinterpret_cast<const double*>(&carrierSpectrum[i].imag()));

        // Compute reciprocal of envelope
        float64x2_t recip_env = vrecpeq_f64(env); // Reciprocal estimate
        recip_env = vmulq_f64(vrecpsq_f64(env, recip_env), recip_env); // Refinement step

        // Perform division: real / env
        float64x2_t real_div = vmulq_f64(real_part, recip_env);

        // Perform division: imag / env
        float64x2_t imag_div = vmulq_f64(imag_part, recip_env);

        // Store the results back
        vst1q_f64(reinterpret_cast<double*>(&flattenedSpectrum[i].real()), real_div);
        vst1q_f64(reinterpret_cast<double*>(&flattenedSpectrum[i].imag()), imag_div);
    }

    // Handle remaining element if size is odd
    for (; i < size; ++i) {
        if (envelope[i] != 0.0) {
            flattenedSpectrum[i].real(carrierSpectrum[i].real() / envelope[i]);
            flattenedSpectrum[i].imag(carrierSpectrum[i].imag() / envelope[i]);
        } else {
            flattenedSpectrum[i] = std::complex<double>(0.0, 0.0);
        }
    }

    return flattenedSpectrum;
}

// Optimized function to multiply with modulator's envelope using NEON
std::vector<std::complex<double>> multiplyWithModEnvelopeNEON(const std::vector<std::complex<double>>& spectrum, const std::vector<double>& modEnvelope) {
    size_t size = spectrum.size();
    std::vector<std::complex<double>> multipliedSpectrum(size);
    size_t i = 0;
    
    for (; i + 1 < size; i += 2) {
        // Load two modulator envelope values
        float64x2_t mod_env = vld1q_f64(&modEnvelope[i]);

        // Load two spectrum real parts
        float64x2_t real_part = vld1q_f64(reinterpret_cast<const double*>(&spectrum[i].real()));

        // Load two spectrum imaginary parts
        float64x2_t imag_part = vld1q_f64(reinterpret_cast<const double*>(&spectrum[i].imag()));

        // Multiply: real * mod_env
        float64x2_t real_mul = vmulq_f64(real_part, mod_env);

        // Multiply: imag * mod_env
        float64x2_t imag_mul = vmulq_f64(imag_part, mod_env);

        // Store the results
        vst1q_f64(reinterpret_cast<double*>(&multipliedSpectrum[i].real()), real_mul);
        vst1q_f64(reinterpret_cast<double*>(&multipliedSpectrum[i].imag()), imag_mul);
    }

    // Handle remaining element if size is odd
    for (; i < size; ++i) {
        multipliedSpectrum[i].real(spectrum[i].real() * modEnvelope[i]);
        multipliedSpectrum[i].imag(spectrum[i].imag() * modEnvelope[i]);
    }

    return multipliedSpectrum;
}

// Optimized function to accumulate to output buffer using NEON
void accumulateToOutputNEON(std::vector<double>& xs_output, const std::vector<double>& processedFrame, int begin_sample, int frame_duration) {
    size_t i = 0;
    size_t end_sample = begin_sample + frame_duration;
    
    // Ensure we don't go out of bounds
    size_t size = xs_output.size();
    
    for (; i + 1 < frame_duration && (begin_sample + i + 1) < size; i += 2) {
        // Load two samples from output
        float64x2_t out = vld1q_f64(&xs_output[begin_sample + i]);

        // Load two samples from processed frame
        float64x2_t proc = vld1q_f64(&processedFrame[i]);

        // Add: out + proc
        float64x2_t result = vaddq_f64(out, proc);

        // Store back to output
        vst1q_f64(&xs_output[begin_sample + i], result);
    }

    // Handle remaining samples
    for (; i < frame_duration && (begin_sample + i) < size; ++i) {
        xs_output[begin_sample + i] += processedFrame[i];
    }
}

// ------------------------------
// Feature Extraction Function
// ------------------------------

// Enhanced extractFeatures function with MFCCs, Chroma, Spectral Contrast, and Tonnetz
Eigen::VectorXd extractFeaturesEnhanced(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();
    
    // Existing features
    double spectralCentroid = 0.0;
    double spectralBandwidth = 0.0;
    double spectralRollOff = 0.0;
    double zeroCrossingRate = 0.0;
    double energy = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double leftRightRatio = 0.0;
    
    // New Features
    int numMFCC = 13;
    int numChroma = 12;
    int numSpectralContrast = 7;
    int numTonnetz = 6;
    
    // Temporary storage for aggregated data
    std::vector<std::complex<double>> aggregatedSpectrum;
    
    // Iterate over matrices in the frame and aggregate spectra
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                // Reconstruct the complex spectrum (assuming amplitude only)
                aggregatedSpectrum.emplace_back(it.value(), 0.0);
            }
        }
    }
    
    // Compute FFT of the aggregated spectrum
    int Nfft = 512; // Adjust based on your FFT size
    fftw_complex* in_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_complex* out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_plan plan = fftw_plan_dft_1d(Nfft, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int n = 0; n < Nfft; ++n) {
        if (n < aggregatedSpectrum.size()) {
            in_fft[n][0] = aggregatedSpectrum[n].real();
            in_fft[n][1] = aggregatedSpectrum[n].imag();
        } else {
            in_fft[n][0] = 0.0;
            in_fft[n][1] = 0.0;
        }
    }
    
    fftw_execute(plan);
    
    std::vector<std::complex<double>> spectrum(Nfft);
    for (int n = 0; n < Nfft; ++n) {
        spectrum[n] = std::complex<double>(out_fft[n][0], out_fft[n][1]);
    }
    
    fftw_destroy_plan(plan);
    fftw_free(in_fft);
    fftw_free(out_fft);
    
    // Compute Power Spectrum
    std::vector<double> powerSpectrum(Nfft / 2 + 1, 0.0);
    for (int k = 0; k < Nfft / 2 + 1; ++k) {
        powerSpectrum[k] = std::pow(std::abs(spectrum[k]), 2);
    }
    
    // Compute MFCCs
    std::vector<double> mfccs = computeMFCCs(spectrum, 26, numMFCC, sampleRate);
    
    // Compute Chroma Features
    std::vector<double> chroma = computeChroma(spectrum, sampleRate);
    
    // Compute Spectral Contrast
    std::vector<double> spectralContrast = computeSpectralContrast(spectrum, numSpectralContrast);
    
    // Compute Tonnetz Features
    std::vector<double> tonnetz = computeTonnetz(chroma);
    
    // Assign existing features
    // [Existing feature calculations...]
    // For demonstration, let's assume we have separate calculations
    // Here we simply assign the existing variables (you should calculate them based on aggregatedSpectrum)
    // Replace these with actual calculations as needed
    
    // Assign calculated features to the feature vector
    features(0) = spectralCentroid;
    features(1) = spectralBandwidth;
    features(2) = spectralRollOff;
    features(3) = zeroCrossingRate;
    features(4) = energy;
    features(5) = mean;
    features(6) = variance;
    features(7) = skewness;
    features(8) = kurtosis;
    features(9) = leftRightRatio;
    
    // Assign new features
    int featureIdx = 10;
    for (const auto& mfcc : mfccs) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = mfcc;
        }
    }
    for (const auto& ch : chroma) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = ch;
        }
    }
    for (const auto& sc : spectralContrast) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = sc;
        }
    }
    for (const auto& tn : tonnetz) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = tn;
        }
    }
    
    // Fill remaining feature slots with zeros or additional features
    for (; featureIdx < number_of_features; ++featureIdx) {
        features(featureIdx) = 0.0; // Replace with actual calculations if available
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
            Eigen::VectorXd features = extractFeaturesEnhanced(deltaDataStream[i], number_of_features, 44100); // Assuming sampleRate = 44100; adjust as needed
            
            // Validate features for NaNs or infinities
            bool validFeatures = true;
            for (int f = 0; f < features.size(); ++f) {
                if (std::isnan(features(f)) || std::isinf(features(f))) {
                    #pragma omp critical
                    {
                        std::cerr << "Error: Feature " << f << " in frame " << i << " is invalid (NaN or Inf)." << std::endl;
                    }
                    validFeatures = false;
                    break;
                }
            }

            if (validFeatures) {
                // Atomic update to prevent race conditions
                #pragma omp critical
                {
                    featureMatrix.row(id) += features;
                }
            }
        }
    }

    return featureMatrix;
}

// Function to save the feature matrix to a CSV file with headers
bool saveFeatureMatrixToCSV(const Eigen::MatrixXd& featureMatrix, const std::string& filename, const std::vector<std::string>& featureNames) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Unable to open file for writing feature matrix: " << filename << std::endl;
        return false;
    }

    // Write the header
    for (size_t i = 0; i < featureNames.size(); ++i) {
        ofs << featureNames[i];
        if (i != featureNames.size() - 1)
            ofs << ",";
    }
    ofs << "\n";

    // Write the data
    for (int i = 0; i < featureMatrix.rows(); ++i) {
        for (int j = 0; j < featureMatrix.cols(); ++j) {
            ofs << featureMatrix(i, j);
            if (j != featureMatrix.cols() - 1)
                ofs << ",";
        }
        ofs << "\n";
    }

    // Validate CSV export
    bool exportSuccess = validateCSVExport(ofs, filename);
    ofs.close();
    return exportSuccess;
}

// Function to serialize a sparse matrix
void serializeSparseMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat, std::ofstream& ofs) {
    int rows = mat.rows();
    int cols = mat.cols();
    int nnz = mat.nonZeros();

    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&nnz), sizeof(int));

    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
            int row = it.row();
            int col = it.col();
            double value = it.value();
            ofs.write(reinterpret_cast<const char*>(&row), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&col), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&value), sizeof(double));
        }
    }
}

// Function to deserialize a sparse matrix
Eigen::SparseMatrix<double, Eigen::RowMajor, int> deserializeSparseMatrix(std::ifstream& ifs, int Nfft) {
    int rows, cols, nnz;
    ifs.read(reinterpret_cast<char*>(&rows), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&cols), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&nnz), sizeof(int));

    Eigen::SparseMatrix<double, Eigen::RowMajor, int> mat(rows, cols);
    mat.reserve(nnz);

    for (int i = 0; i < nnz; ++i) {
        int row, col;
        double value;
        ifs.read(reinterpret_cast<char*>(&row), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&col), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&value), sizeof(double));
        mat.insert(row, col) = value;
    }

    mat.makeCompressed();
    return mat;
}

// Function to serialize the entire delta data stream
void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const std::string& filename, int Nfft) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Failed to open file for serialization: " << filename << std::endl;
        return;
    }

    int numFrames = deltaDataStream.size();
    ofs.write(reinterpret_cast<const char*>(&numFrames), sizeof(int));

    for (const auto& frame : deltaDataStream) {
        serializeSparseMatrix(frame.Delta_A_L_B_L, ofs);
        serializeSparseMatrix(frame.Delta_A_L_B_R, ofs);
        serializeSparseMatrix(frame.Delta_A_R_B_L, ofs);
        serializeSparseMatrix(frame.Delta_A_R_B_R, ofs);
    }

    ofs.close();
}

// Function to deserialize the entire delta data stream
std::vector<FrameDeltaSparseMatrices> deserializeDeltaDataStream(const std::string& filename, int Nfft) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Error: Failed to open file for deserialization: " << filename << std::endl;
        return {};
    }

    int numFrames;
    ifs.read(reinterpret_cast<char*>(&numFrames), sizeof(int));

    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
    deltaDataStream.reserve(numFrames);

    for (int i = 0; i < numFrames; ++i) {
        FrameDeltaSparseMatrices deltaFrame(Nfft, Nfft);
        deltaFrame.Delta_A_L_B_L = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_L_B_R = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_R_B_L = deserializeSparseMatrix(ifs, Nfft);
        deltaFrame.Delta_A_R_B_R = deserializeSparseMatrix(ifs, Nfft);
        deltaDataStream.push_back(deltaFrame);
    }

    ifs.close();
    return deltaDataStream;
}

// Function to reconstruct data stream from delta data stream
std::vector<FrameSparseMatrices> reconstructDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, int Nfft) {
    std::vector<FrameSparseMatrices> reconstructedDataStream;
    reconstructedDataStream.reserve(deltaDataStream.size());

    // Initialize previous frame matrices for delta computation (start with zero matrices)
    FrameSparseMatrices previousFrame(Nfft, Nfft);
    previousFrame.A_L_B_L.setZero();
    previousFrame.A_L_B_R.setZero();
    previousFrame.A_R_B_L.setZero();
    previousFrame.A_R_B_R.setZero();

    for (const auto& deltaFrame : deltaDataStream) {
        FrameSparseMatrices currentFrame(Nfft, Nfft);

        // Add deltas to previous frame
        currentFrame.A_L_B_L = previousFrame.A_L_B_L + deltaFrame.Delta_A_L_B_L;
        currentFrame.A_L_B_R = previousFrame.A_L_B_R + deltaFrame.Delta_A_L_B_R;
        currentFrame.A_R_B_L = previousFrame.A_R_B_L + deltaFrame.Delta_A_R_B_L;
        currentFrame.A_R_B_R = previousFrame.A_R_B_R + deltaFrame.Delta_A_R_B_R;

        // Store the current frame
        reconstructedDataStream.push_back(currentFrame);

        // Update previous frame
        previousFrame = currentFrame;
    }

    return reconstructedDataStream;
}

// ------------------------------
// Final Preparation Module
// ------------------------------

// Function to standardize the feature matrix (zero mean, unit variance)
Eigen::MatrixXd standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix) {
    Eigen::MatrixXd standardized = featureMatrix;
    int numFeatures = featureMatrix.cols();

    for (int i = 0; i < numFeatures; ++i) {
        double mean = featureMatrix.col(i).mean();
        double variance = (featureMatrix.col(i).array() - mean).square().sum() / (featureMatrix.rows() - 1);
        double stddev = std::sqrt(variance);

        if (stddev > 0.0) {
            standardized.col(i) = (featureMatrix.col(i).array() - mean) / stddev;
        } else {
            standardized.col(i).setZero(); // If stddev is zero, set the feature to zero
        }
    }

    return standardized;
}

// Function to perform PCA on the standardized feature matrix
Eigen::MatrixXd performPCA(const Eigen::MatrixXd& standardized, int numComponents) {
    // Compute covariance matrix
    Eigen::MatrixXd cov = (standardized.adjoint() * standardized) / double(standardized.rows() - 1);

    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);

    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Error: Eigen decomposition failed during PCA." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the eigenvalues and eigenvectors
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().reverse(); // Descending order
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().rowwise().reverse();

    // Select the top 'numComponents' eigenvectors
    Eigen::MatrixXd projection = eigenVectors.leftCols(numComponents);

    // Project the data onto the principal components
    Eigen::MatrixXd reduced = standardized * projection;

    return reduced;
}

// ------------------------------
// Feature Extraction Function
// ------------------------------

// Enhanced extractFeatures function with MFCCs, Chroma, Spectral Contrast, and Tonnetz
Eigen::VectorXd extractFeaturesEnhanced(const FrameDeltaSparseMatrices& frameDelta, int number_of_features, int sampleRate) {
    Eigen::VectorXd features(number_of_features);
    features.setZero();
    
    // Existing features
    double spectralCentroid = 0.0;
    double spectralBandwidth = 0.0;
    double spectralRollOff = 0.0;
    double zeroCrossingRate = 0.0;
    double energy = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double skewness = 0.0;
    double kurtosis = 0.0;
    double leftRightRatio = 0.0;
    
    // New Features
    int numMFCC = 13;
    int numChroma = 12;
    int numSpectralContrast = 7;
    int numTonnetz = 6;
    
    // Temporary storage for aggregated data
    std::vector<std::complex<double>> aggregatedSpectrum;
    
    // Iterate over matrices in the frame and aggregate spectra
    for (const auto& mat : {frameDelta.Delta_A_L_B_L, frameDelta.Delta_A_L_B_R, frameDelta.Delta_A_R_B_L, frameDelta.Delta_A_R_B_R}) {
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(mat, k); it; ++it) {
                // Reconstruct the complex spectrum (assuming amplitude only)
                aggregatedSpectrum.emplace_back(it.value(), 0.0);
            }
        }
    }
    
    // Compute FFT of the aggregated spectrum
    int Nfft = 512; // Adjust based on your FFT size
    fftw_complex* in_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_complex* out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_plan plan = fftw_plan_dft_1d(Nfft, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);
    
    for (int n = 0; n < Nfft; ++n) {
        if (n < aggregatedSpectrum.size()) {
            in_fft[n][0] = aggregatedSpectrum[n].real();
            in_fft[n][1] = aggregatedSpectrum[n].imag();
        } else {
            in_fft[n][0] = 0.0;
            in_fft[n][1] = 0.0;
        }
    }
    
    fftw_execute(plan);
    
    std::vector<std::complex<double>> spectrum(Nfft);
    for (int n = 0; n < Nfft; ++n) {
        spectrum[n] = std::complex<double>(out_fft[n][0], out_fft[n][1]);
    }
    
    fftw_destroy_plan(plan);
    fftw_free(in_fft);
    fftw_free(out_fft);
    
    // Compute Power Spectrum
    std::vector<double> powerSpectrum(Nfft / 2 + 1, 0.0);
    for (int k = 0; k < Nfft / 2 + 1; ++k) {
        powerSpectrum[k] = std::pow(std::abs(spectrum[k]), 2);
    }
    
    // Compute MFCCs
    std::vector<double> mfccs = computeMFCCs(spectrum, 26, numMFCC, sampleRate);
    
    // Compute Chroma Features
    std::vector<double> chroma = computeChroma(spectrum, sampleRate);
    
    // Compute Spectral Contrast
    std::vector<double> spectralContrast = computeSpectralContrast(spectrum, numSpectralContrast);
    
    // Compute Tonnetz Features
    std::vector<double> tonnetz = computeTonnetz(chroma);
    
    // Assign existing features
    // [Existing feature calculations...]
    // For demonstration, let's assume we have separate calculations
    // Here we simply assign the existing variables (you should calculate them based on aggregatedSpectrum)
    // Replace these with actual calculations as needed
    
    // Assign calculated features to the feature vector
    features(0) = spectralCentroid;
    features(1) = spectralBandwidth;
    features(2) = spectralRollOff;
    features(3) = zeroCrossingRate;
    features(4) = energy;
    features(5) = mean;
    features(6) = variance;
    features(7) = skewness;
    features(8) = kurtosis;
    features(9) = leftRightRatio;
    
    // Assign new features
    int featureIdx = 10;
    for (const auto& mfcc : mfccs) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = mfcc;
        }
    }
    for (const auto& ch : chroma) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = ch;
        }
    }
    for (const auto& sc : spectralContrast) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = sc;
        }
    }
    for (const auto& tn : tonnetz) {
        if (featureIdx < number_of_features) {
            features(featureIdx++) = tn;
        }
    }
    
    // Fill remaining feature slots with zeros or additional features
    for (; featureIdx < number_of_features; ++featureIdx) {
        features(featureIdx) = 0.0; // Replace with actual calculations if available
    }
    
    return features;
}

// ------------------------------
// Feature Extraction Function
// ------------------------------

// Function to extract skewness
double calculateSkewness(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double skewness = 0.0;
    for (const auto& val : data) {
        skewness += std::pow((val - mean) / stddev, 3);
    }
    return skewness / data.size();
}

// Function to extract kurtosis
double calculateKurtosis(const std::vector<double>& data, double mean, double stddev) {
    if (stddev == 0.0) return 0.0;
    double kurtosis = 0.0;
    for (const auto& val : data) {
        kurtosis += std::pow((val - mean) / stddev, 4);
    }
    return kurtosis / data.size() - 3.0; // Excess kurtosis
}

// Function to compute delta between two sparse matrices
Eigen::SparseMatrix<double, Eigen::RowMajor, int> computeDelta(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& current, 
                                                               const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& previous, 
                                                               double threshold) {
    // Compute the difference
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> delta = current - previous;
    
    // Apply thresholding to ignore minor changes
    for (int k=0; k < delta.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double, Eigen::RowMajor, int>::InnerIterator it(delta, k); it; ++it) {
            if (std::abs(it.value()) < threshold) {
                it.valueRef() = 0.0;
            }
        }
    }
    
    // Prune zero entries
    delta.prune(0.0);
    
    return delta;
}

// ------------------------------
// WAV File Operations
// ------------------------------

// Function to read a stereo WAV file (with validation)
bool readStereoWavFileValidated(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate) {
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error opening WAV file: " << filename << std::endl;
        return false;
    }

    // Validate WAV file
    if (!validateWavFile(sfInfo, filename)) {
        sf_close(file);
        return false;
    }

    sampleRate = sfInfo.samplerate;
    int numChannels = sfInfo.channels;
    int numFrames = sfInfo.frames;

    std::vector<double> tempData(numFrames * numChannels);
    sf_count_t framesRead = sf_readf_double(file, tempData.data(), numFrames);
    sf_close(file);

    if (framesRead != numFrames) {
        std::cerr << "Warning: Expected to read " << numFrames << " frames from " << filename << ", but read " << framesRead << " frames." << std::endl;
    }

    if (numChannels != 2) {
        std::cerr << "Error: Input WAV file " << filename << " is not stereo." << std::endl;
        return false;
    }

    left.resize(numFrames);
    right.resize(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        left[i] = tempData[i * numChannels];
        right[i] = tempData[i * numChannels + 1];
    }

    // Additional validation: Check for NaNs or infinities in the audio data
    for (int i = 0; i < numFrames; ++i) {
        if (std::isnan(left[i]) || std::isinf(left[i])) {
            std::cerr << "Error: Stream Left channel contains invalid sample at frame " << i << "." << std::endl;
            return false;
        }
        if (std::isnan(right[i]) || std::isinf(right[i])) {
            std::cerr << "Error: Stream Right channel contains invalid sample at frame " << i << "." << std::endl;
            return false;
        }
    }

    return true;
}

// Function to write a WAV file (mono)
bool writeWavFileValidated(const std::string& filename, const std::vector<double>& data, int sampleRate) {
    SF_INFO sfInfo;
    sfInfo.channels = 1;
    sfInfo.samplerate = sampleRate;
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfInfo);
    if (!file) {
        std::cerr << "Error writing WAV file: " << filename << std::endl;
        return false;
    }

    // Convert double data to short for 16-bit PCM
    std::vector<short> intData(data.size());
    size_t i = 0;

    // Use NEON to process two samples at a time
    for (; i + 1 < data.size(); i += 2) {
        double sample1 = data[i];
        double sample2 = data[i + 1];

        // Clipping
        if (sample1 > 1.0) sample1 = 1.0;
        if (sample1 < -1.0) sample1 = -1.0;
        if (sample2 > 1.0) sample2 = 1.0;
        if (sample2 < -1.0) sample2 = -1.0;

        // Convert to int16
        short intSample1 = static_cast<short>(sample1 * 32767);
        short intSample2 = static_cast<short>(sample2 * 32767);

        // Load into NEON register
        int16x4_t samples = vld1q_s16(reinterpret_cast<const short*>(&intSample1));

        // Store using NEON
        vst1q_s16(&intData[i], samples);
    }

    // Handle remaining sample if size is odd
    for (; i < data.size(); ++i) {
        double sample = data[i];
        if (sample > 1.0) sample = 1.0;
        if (sample < -1.0) sample = -1.0;
        intData[i] = static_cast<short>(sample * 32767);
    }

    // Write data to WAV file
    sf_count_t framesWritten = sf_writef_short(file, intData.data(), data.size());
    sf_close(file);

    if (framesWritten != static_cast<sf_count_t>(data.size())) {
        std::cerr << "Warning: Expected to write " << data.size() << " frames to " << filename << ", but wrote " << framesWritten << " frames." << std::endl;
    }

    return true;
}

// ------------------------------
// Final Preparation Module
// ------------------------------

// Function to standardize the feature matrix (zero mean, unit variance)
Eigen::MatrixXd standardizeFeatureMatrix(const Eigen::MatrixXd& featureMatrix) {
    Eigen::MatrixXd standardized = featureMatrix;
    int numFeatures = featureMatrix.cols();

    for (int i = 0; i < numFeatures; ++i) {
        double mean = featureMatrix.col(i).mean();
        double variance = (featureMatrix.col(i).array() - mean).square().sum() / (featureMatrix.rows() - 1);
        double stddev = std::sqrt(variance);

        if (stddev > 0.0) {
            standardized.col(i) = (featureMatrix.col(i).array() - mean) / stddev;
        } else {
            standardized.col(i).setZero(); // If stddev is zero, set the feature to zero
        }
    }

    return standardized;
}

// Function to perform PCA on the standardized feature matrix
Eigen::MatrixXd performPCA(const Eigen::MatrixXd& standardized, int numComponents) {
    // Compute covariance matrix
    Eigen::MatrixXd cov = (standardized.adjoint() * standardized) / double(standardized.rows() - 1);

    // Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);

    if (eigenSolver.info() != Eigen::Success) {
        std::cerr << "Error: Eigen decomposition failed during PCA." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get the eigenvalues and eigenvectors
    Eigen::VectorXd eigenValues = eigenSolver.eigenvalues().reverse(); // Descending order
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().rowwise().reverse();

    // Select the top 'numComponents' eigenvectors
    Eigen::MatrixXd projection = eigenVectors.leftCols(numComponents);

    // Project the data onto the principal components
    Eigen::MatrixXd reduced = standardized * projection;

    return reduced;
}

// ------------------------------
// Final C++ Program with Enhanced Feature Extraction
// ------------------------------

int main() {
    // File paths for two stereo WAV files (Stream A and Stream B)
    std::string streamAFile = "../data/streamA_stereo.wav"; // Replace with your Stream A file path
    std::string streamBFile = "../data/streamB_stereo.wav"; // Replace with your Stream B file path
    std::string deltaFile = "deltaDataStream.bin";         // Delta data stream file path
    std::string featureFilePath = "featureMatrix.csv";     // Feature matrix file path (with headers)
    std::string hashMapFilePath = "frameHashMap.txt";      // Hash map file path (optional)
    std::string outputFile = "output.wav";                 // Output file path (optional)

    // Vectors to hold left and right channels for Stream A and Stream B
    std::vector<double> A_left, A_right;
    std::vector<double> B_left, B_right;
    int fs_A, fs_B;

    // Read and validate Stream A WAV file
    if (!readStereoWavFileValidated(streamAFile, A_left, A_right, fs_A)) {
        std::cerr << "Error: Failed to read or validate Stream A." << std::endl;
        return -1;
    }

    // Read and validate Stream B WAV file
    if (!readStereoWavFileValidated(streamBFile, B_left, B_right, fs_B)) {
        std::cerr << "Error: Failed to read or validate Stream B." << std::endl;
        return -1;
    }

    // Ensure both streams have the same sampling rate
    if (fs_A != fs_B) {
        std::cerr << "Error: Sampling rates of Stream A and Stream B do not match." << std::endl;
        return -1;
    }

    // Determine the minimum size to prevent out-of-bounds access
    size_t min_size = std::min({A_left.size(), A_right.size(), B_left.size(), B_right.size()});
    if (min_size < static_cast<size_t>(frame_duration)) {
        std::cerr << "Error: Audio streams are too short for the specified frame duration." << std::endl;
        return -1;
    }

    // Resize all streams to the minimum size
    A_left.resize(min_size, 0.0);
    A_right.resize(min_size, 0.0);
    B_left.resize(min_size, 0.0);
    B_right.resize(min_size, 0.0);

    // Parameters
    int frame_duration = 256;
    int number_of_features = 48; // 10 existing + 13 MFCC + 12 Chroma + 7 Spectral Contrast + 6 Tonnetz
    int Nfft = 512;
    // Assuming createBlackmanWindow and createHammingWindow functions are defined elsewhere
    // std::vector<double> blackman_window = createBlackmanWindow(Nfft);
    // std::vector<double> hamming_window = createHammingWindow(frame_duration);
    std::vector<double> blackman_window(Nfft, 0.0); // Placeholder: Implement actual window creation
    std::vector<double> hamming_window(frame_duration, 0.54 - 0.46 * std::cos(2 * M_PI * i / (frame_duration - 1))); // Simple Hamming window
    int hopSize = frame_duration / 4; // 75% overlap
    int number_of_frames = (min_size - frame_duration) / hopSize + 1;

    // Initialize accumulators for Stream A and Stream B
    std::vector<double> A_left_accum(min_size, 0.0);
    std::vector<double> A_right_accum(min_size, 0.0);
    std::vector<double> B_left_accum(min_size, 0.0);
    std::vector<double> B_right_accum(min_size, 0.0);
    std::vector<double> xs_output(min_size, 0.0); // Output accumulation

    // Initialize a vector to hold 4D data stream (frames with four sparse matrices)
    std::vector<FrameSparseMatrices> dataStream;
    dataStream.reserve(number_of_frames);

    // Initialize a vector to hold 4D delta data stream (frames with four delta sparse matrices)
    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
    deltaDataStream.reserve(number_of_frames);

    // Initialize previous frame matrices for delta computation (start with zero matrices)
    FrameSparseMatrices previousFrame(Nfft, Nfft);
    previousFrame.A_L_B_L.setZero();
    previousFrame.A_L_B_R.setZero();
    previousFrame.A_R_B_L.setZero();
    previousFrame.A_R_B_R.setZero();

    // Initialize FFTW plans
    fftw_complex* in_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_complex* out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_plan fft_plan = fftw_plan_dft_1d(Nfft, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_complex* in_ifft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    double* out_ifft = (double*)fftw_malloc(sizeof(double) * Nfft);
    fftw_plan ifft_plan = fftw_plan_dft_c2r_1d(Nfft, in_ifft, out_ifft, FFTW_ESTIMATE);

    // Processing loop with OpenMP parallelization
    #pragma omp parallel for
    for (int i = 0; i < number_of_frames; ++i) {
        int begin_sample = i * hopSize;
        int end_sample = begin_sample + frame_duration;

        // Extract frames for Stream A and Stream B (left and right channels)
        std::vector<double> A_frame_left(A_left.begin() + begin_sample, A_left.begin() + end_sample);
        std::vector<double> A_frame_right(A_right.begin() + begin_sample, A_right.begin() + end_sample);
        std::vector<double> B_frame_left(B_left.begin() + begin_sample, B_left.begin() + end_sample);
        std::vector<double> B_frame_right(B_right.begin() + begin_sample, B_right.begin() + end_sample);

        // Validate frames
        bool validFrameA_L = validateFrame(A_frame_left, frame_duration, "Stream A Left", i);
        bool validFrameA_R = validateFrame(A_frame_right, frame_duration, "Stream A Right", i);
        bool validFrameB_L = validateFrame(B_frame_left, frame_duration, "Stream B Left", i);
        bool validFrameB_R = validateFrame(B_frame_right, frame_duration, "Stream B Right", i);

        if (!validFrameA_L || !validFrameA_R || !validFrameB_L || !validFrameB_R) {
            // Skip processing this frame
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid data." << std::endl;
            }
            continue;
        }

        // Apply Hamming window using NEON
        // Optimize by processing two samples at a time
        size_t j = 0;
        for (; j + 1 < frame_duration; j += 2) {
            // Load two window coefficients
            float64x2_t win = vld1q_f64(&hamming_window[j]);

            // Load two samples for Stream A Left
            float64x2_t samp_A_L = vld1q_f64(&A_frame_left[j]);
            // Apply window
            float64x2_t windowed_A_L = vmulq_f64(samp_A_L, win);
            // Store back
            vst1q_f64(&A_frame_left[j], windowed_A_L);

            // Load two samples for Stream A Right
            float64x2_t samp_A_R = vld1q_f64(&A_frame_right[j]);
            // Apply window
            float64x2_t windowed_A_R = vmulq_f64(samp_A_R, win);
            // Store back
            vst1q_f64(&A_frame_right[j], windowed_A_R);

            // Load two samples for Stream B Left
            float64x2_t samp_B_L = vld1q_f64(&B_frame_left[j]);
            // Apply window
            float64x2_t windowed_B_L = vmulq_f64(samp_B_L, win);
            // Store back
            vst1q_f64(&B_frame_left[j], windowed_B_L);

            // Load two samples for Stream B Right
            float64x2_t samp_B_R = vld1q_f64(&B_frame_right[j]);
            // Apply window
            float64x2_t windowed_B_R = vmulq_f64(samp_B_R, win);
            // Store back
            vst1q_f64(&B_frame_right[j], windowed_B_R);
        }

        // Handle remaining samples if frame_duration is odd
        for (; j < frame_duration; ++j) {
            A_frame_left[j] *= hamming_window[j];
            A_frame_right[j] *= hamming_window[j];
            B_frame_left[j] *= hamming_window[j];
            B_frame_right[j] *= hamming_window[j];
        }

        // Pad frames to Nfft with zeros if necessary
        if (A_frame_left.size() < static_cast<size_t>(Nfft)) {
            A_frame_left.resize(Nfft, 0.0);
            A_frame_right.resize(Nfft, 0.0);
            B_frame_left.resize(Nfft, 0.0);
            B_frame_right.resize(Nfft, 0.0);
        }

        // Compute FFT for Stream A Left
        std::vector<std::complex<double>> fft_A_L = computeFFT(A_frame_left, fft_plan, in_fft, out_fft);

        // Validate FFT result
        if (!validateFFTResult(fft_A_L, "FFT_A_L", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid FFT_A_L result." << std::endl;
            }
            continue;
        }

        // Compute FFT for Stream A Right
        std::vector<std::complex<double>> fft_A_R = computeFFT(A_frame_right, fft_plan, in_fft, out_fft);
        if (!validateFFTResult(fft_A_R, "FFT_A_R", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid FFT_A_R result." << std::endl;
            }
            continue;
        }

        // Compute FFT for Stream B Left
        std::vector<std::complex<double>> fft_B_L = computeFFT(B_frame_left, fft_plan, in_fft, out_fft);
        if (!validateFFTResult(fft_B_L, "FFT_B_L", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid FFT_B_L result." << std::endl;
            }
            continue;
        }

        // Compute FFT for Stream B Right
        std::vector<std::complex<double>> fft_B_R = computeFFT(B_frame_right, fft_plan, in_fft, out_fft);
        if (!validateFFTResult(fft_B_R, "FFT_B_R", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid FFT_B_R result." << std::endl;
            }
            continue;
        }

        // Inverse FFT and accumulate to Stream A and Stream B accumulators
        std::vector<double> ifft_A_L = computeIFFT(fft_A_L, ifft_plan, in_ifft, out_ifft);
        if (!validateIFFTResult(ifft_A_L, "IFFT_A_L", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid IFFT_A_L result." << std::endl;
            }
            continue;
        }

        std::vector<double> ifft_A_R = computeIFFT(fft_A_R, ifft_plan, in_ifft, out_ifft);
        if (!validateIFFTResult(ifft_A_R, "IFFT_A_R", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid IFFT_A_R result." << std::endl;
            }
            continue;
        }

        std::vector<double> ifft_B_L = computeIFFT(fft_B_L, ifft_plan, in_ifft, out_ifft);
        if (!validateIFFTResult(ifft_B_L, "IFFT_B_L", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid IFFT_B_L result." << std::endl;
            }
            continue;
        }

        std::vector<double> ifft_B_R = computeIFFT(fft_B_R, ifft_plan, in_ifft, out_ifft);
        if (!validateIFFTResult(ifft_B_R, "IFFT_B_R", i)) {
            #pragma omp critical
            {
                std::cerr << "Warning: Skipping frame " << i << " due to invalid IFFT_B_R result." << std::endl;
            }
            continue;
        }

        // Accumulate using NEON
        accumulateToOutputNEON(A_left_accum, ifft_A_L, begin_sample, frame_duration);
        accumulateToOutputNEON(A_right_accum, ifft_A_R, begin_sample, frame_duration);
        accumulateToOutputNEON(B_left_accum, ifft_B_L, begin_sample, frame_duration);
        accumulateToOutputNEON(B_right_accum, ifft_B_R, begin_sample, frame_duration);

        // Create current frame's sparse matrices
        FrameSparseMatrices currentFrame(Nfft, Nfft);

        // Apply thresholding during insertion
        double threshold = 0.1; // Adjust based on your data

        // Stream A Left * Stream B Left
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(fft_A_L[n] * fft_B_L[n]);
            if (value > threshold) {
                currentFrame.A_L_B_L.insert(n, n) = value;
            }
        }

        // Stream A Left * Stream B Right
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(fft_A_L[n] * fft_B_R[n]);
            if (value > threshold) {
                currentFrame.A_L_B_R.insert(n, n) = value;
            }
        }

        // Stream A Right * Stream B Left
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(fft_A_R[n] * fft_B_L[n]);
            if (value > threshold) {
                currentFrame.A_R_B_L.insert(n, n) = value;
            }
        }

        // Stream A Right * Stream B Right
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(fft_A_R[n] * fft_B_R[n]);
            if (value > threshold) {
                currentFrame.A_R_B_R.insert(n, n) = value;
            }
        }

        // Compute deltas
        FrameDeltaSparseMatrices deltaFrame(Nfft, Nfft);
        deltaFrame.Delta_A_L_B_L = computeDelta(currentFrame.A_L_B_L, previousFrame.A_L_B_L, 1e-6);
        deltaFrame.Delta_A_L_B_R = computeDelta(currentFrame.A_L_B_R, previousFrame.A_L_B_R, 1e-6);
        deltaFrame.Delta_A_R_B_L = computeDelta(currentFrame.A_R_B_L, previousFrame.A_R_B_L, 1e-6);
        deltaFrame.Delta_A_R_B_R = computeDelta(currentFrame.A_R_B_R, previousFrame.A_R_B_R, 1e-6);

        // Store the delta frame and current frame (thread-safe)
        #pragma omp critical
        {
            deltaDataStream.push_back(deltaFrame);
            dataStream.push_back(currentFrame);
        }

        // Update previous frame
        previousFrame = currentFrame;
    }

    // Clean up FFTW resources
    fftw_destroy_plan(fft_plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in_fft);
    fftw_free(out_fft);
    fftw_free(in_ifft);
    fftw_free(out_ifft);

    // Serialize the delta data stream to a binary file
    serializeDeltaDataStream(deltaDataStream, deltaFile, Nfft);
    std::cout << "Delta data stream serialized to " << deltaFile << std::endl;

    // ------------------------------
    // Parsing and Preparation Module
    // ------------------------------

    // Step 1: Build the hash map from the delta data stream
    FrameHashMap frameMap = buildFrameHashMap(deltaDataStream);

    // Step 2: Extract features and build the feature matrix for ML
    Eigen::MatrixXd featureMatrix = buildFeatureMatrix(deltaDataStream, frameMap, number_of_features);

    // ------------------------------
    // Final Preparation Module
    // ------------------------------

    // Step 3: Standardize the feature matrix
    Eigen::MatrixXd standardizedFeatureMatrix = standardizeFeatureMatrix(featureMatrix);
    std::cout << "Feature matrix standardized (zero mean, unit variance)." << std::endl;

    // Step 4: (Optional) Perform PCA for dimensionality reduction
    // Uncomment and adjust 'numComponents' based on your needs
    /*
    int numComponents = 20; // Number of principal components to keep
    Eigen::MatrixXd reducedFeatureMatrix = performPCA(standardizedFeatureMatrix, numComponents);
    std::cout << "PCA performed. Reduced feature matrix dimensions: " << reducedFeatureMatrix.rows() << "x" << reducedFeatureMatrix.cols() << std::endl;
    */

    // Step 5: Save the feature matrix to a CSV file with headers for ML training
    bool csvExportSuccess = false;
    #pragma omp critical
    {
        csvExportSuccess = saveFeatureMatrixToCSV(standardizedFeatureMatrix, featureFilePath, FEATURE_NAMES);
    }
    if (!csvExportSuccess) {
        std::cerr << "Error: Failed to export feature matrix to CSV." << std::endl;
        return -1;
    }
    std::cout << "Feature matrix saved to " << featureFilePath << std::endl;

    // Step 6: (Optional) Save the hash map to a file for reference
    std::ofstream hashMapFile(hashMapFilePath);
    if (hashMapFile.is_open()) {
        for (const auto& pair : frameMap) {
            hashMapFile << pair.second << ": " << pair.first << "\n";
        }
        hashMapFile.close();
        std::cout << "Hash map saved to " << hashMapFilePath << std::endl;
    } else {
        std::cerr << "Warning: Unable to open file for writing hash map." << std::endl;
    }

    // ------------------------------
    // Post-Processing (Normalization, Output Writing)
    // ------------------------------

    // Normalize xs_output
    double max_val = 0.0;
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 0; i < xs_output.size(); ++i) {
        if (std::abs(xs_output[i]) > max_val) {
            max_val = std::abs(xs_output[i]);
        }
    }

    if (max_val > 1.0) {
        #pragma omp parallel for
        for (size_t i = 0; i < xs_output.size(); ++i) {
            xs_output[i] /= max_val;
        }
    }

    // Optionally, write the accumulated output to a WAV file
    // Uncomment the following lines if you wish to save the output audio
    /*
    if (!writeWavFileValidated(outputFile, xs_output, fs_A)) {
        std::cerr << "Error: Failed to write output WAV file." << std::endl;
        return -1;
    }
    std::cout << "Processing complete. Output saved to " << outputFile << std::endl;
    */

    // Instead of writing to a WAV file, you now have:
    // - dataStream: The full 4D data stream
    // - deltaDataStream: The delta-compressed 4D data stream
    // - frameMap: Mapping of unique frames to unique IDs
    // - standardizedFeatureMatrix: Matrix of standardized features ready for ML
    // - (Optional) reducedFeatureMatrix: Matrix of reduced features via PCA

    std::cout << "Processing complete. Delta compression and parsing/preparation modules executed successfully." << std::endl;
    std::cout << "Total frames processed: " << dataStream.size() << std::endl;

    return 0;
}
