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
// Function Declarations
// ------------------------------

// WAV file operations
bool readStereoWavFile(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate);
bool writeWavFile(const std::string& filename, const std::vector<double>& data, int sampleRate);

// Noise Generation
void generateWhiteNoiseNEON(std::vector<double>& carrier, double amplitude = 0.7);

// Window Functions
std::vector<double> createBlackmanWindow(int size);
std::vector<double> createHammingWindow(int size);

// FFT Operations
std::vector<std::complex<double>> computeFFT(const std::vector<double>& frame, fftw_plan& plan, fftw_complex* in, fftw_complex* out);
std::vector<double> computeIFFT(const std::vector<std::complex<double>>& spectrum, fftw_plan& plan, fftw_complex* in, double* out);

// Interpolation
std::vector<double> linearInterpolate(const std::vector<double>& input, int targetSize);

// NEON Optimized Operations
std::vector<std::complex<double>> flattenCarrierSpectrumNEON(const std::vector<std::complex<double>>& carrierSpectrum, const std::vector<double>& envelope);
std::vector<std::complex<double>> multiplyWithModEnvelopeNEON(const std::vector<std::complex<double>>& spectrum, const std::vector<double>& modEnvelope);
void accumulateToOutputNEON(std::vector<double>& xs_output, const std::vector<double>& processedFrame, int begin_sample, int frame_duration);

// Delta Compression
Eigen::SparseMatrix<double, Eigen::RowMajor, int> computeDelta(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& current, 
                                                               const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& previous, 
                                                               double threshold = 1e-6);

// Serialization and Deserialization
void serializeSparseMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat, std::ofstream& ofs);
Eigen::SparseMatrix<double, Eigen::RowMajor, int> deserializeSparseMatrix(std::ifstream& ifs);
void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const std::string& filename);
std::vector<FrameDeltaSparseMatrices> deserializeDeltaDataStream(const std::string& filename);

// Reconstruct Data Stream from Deltas
std::vector<FrameSparseMatrices> reconstructDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, int Nfft);

// ------------------------------
// Function Implementations
// ------------------------------

// Function to read a stereo WAV file and split into left and right channels
bool readStereoWavFile(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate) {
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        std::cerr << "Error opening WAV file: " << filename << std::endl;
        return false;
    }

    sampleRate = sfInfo.samplerate;
    int numChannels = sfInfo.channels;
    int numFrames = sfInfo.frames;

    std::vector<double> tempData(numFrames * numChannels);
    sf_readf_double(file, tempData.data(), numFrames);
    sf_close(file);

    if (numChannels != 2) {
        std::cerr << "Input WAV file is not stereo: " << filename << std::endl;
        return false;
    }

    left.resize(numFrames);
    right.resize(numFrames);
    for (int i = 0; i < numFrames; ++i) {
        left[i] = tempData[i * numChannels];
        right[i] = tempData[i * numChannels + 1];
    }

    return true;
}

// Function to write a WAV file (mono)
bool writeWavFile(const std::string& filename, const std::vector<double>& data, int sampleRate) {
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

    sf_writef_short(file, intData.data(), data.size());
    sf_close(file);

    return true;
}

// Optimized function to generate white noise using NEON
void generateWhiteNoiseNEON(std::vector<double>& carrier, double amplitude = 0.7) {
    size_t size = carrier.size();
    size_t i = 0;
    
    // Define a random number generator for double precision
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    // Process 2 samples at a time (NEON supports 2 doubles per vector)
    for (; i + 1 < size; i += 2) {
        // Generate two random samples
        double sample1 = amplitude * dis(gen);
        double sample2 = amplitude * dis(gen);
        
        // Load two samples into NEON register
        float64x2_t noise = vld1q_f64(&sample1);
        
        // Store them back to the carrier
        vst1q_f64(&carrier[i], noise);
    }
    
    // Handle remaining sample if size is odd
    for (; i < size; ++i) {
        carrier[i] = amplitude * dis(gen);
    }
}

// Function to create Blackman window
std::vector<double> createBlackmanWindow(int size) {
    std::vector<double> window(size);
    for (int n = 0; n < size; ++n) {
        window[n] = 0.42 - 0.5 * cos((2 * M_PI * n) / (size - 1)) + 0.08 * cos((4 * M_PI * n) / (size - 1));
    }
    return window;
}

// Function to create Hamming window
std::vector<double> createHammingWindow(int size) {
    std::vector<double> window(size);
    for (int n = 0; n < size; ++n) {
        window[n] = 0.54 - 0.46 * cos((2 * M_PI * n) / (size - 1));
    }
    return window;
}

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
Eigen::SparseMatrix<double, Eigen::RowMajor, int> deserializeSparseMatrix(std::ifstream& ifs) {
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
void serializeDeltaDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open file for serialization: " << filename << std::endl;
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
std::vector<FrameDeltaSparseMatrices> deserializeDeltaDataStream(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open file for deserialization: " << filename << std::endl;
        return {};
    }
    
    int numFrames;
    ifs.read(reinterpret_cast<char*>(&numFrames), sizeof(int));
    
    std::vector<FrameDeltaSparseMatrices> deltaDataStream;
    deltaDataStream.reserve(numFrames);
    
    for (int i = 0; i < numFrames; ++i) {
        FrameDeltaSparseMatrices deltaFrame(Nfft, Nfft);
        deltaFrame.Delta_A_L_B_L = deserializeSparseMatrix(ifs);
        deltaFrame.Delta_A_L_B_R = deserializeSparseMatrix(ifs);
        deltaFrame.Delta_A_R_B_L = deserializeSparseMatrix(ifs);
        deltaFrame.Delta_A_R_B_R = deserializeSparseMatrix(ifs);
        deltaDataStream.push_back(deltaFrame);
    }
    
    ifs.close();
    return deltaDataStream;
}

// Function to reconstruct data stream from delta data stream
std::vector<FrameSparseMatrices> reconstructDataStream(const std::vector<FrameDeltaSparseMatrices>& deltaDataStream, int Nfft) {
    std::vector<FrameSparseMatrices> reconstructedDataStream;
    reconstructedDataStream.reserve(deltaDataStream.size());
    
    // Initialize previous frame as zero matrices
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
// Main Function
// ------------------------------

int main() {
    // File paths for two stereo WAV files (Stream A and Stream B)
    std::string streamAFile = "../data/streamA_stereo.wav"; // Replace with your Stream A file path
    std::string streamBFile = "../data/streamB_stereo.wav"; // Replace with your Stream B file path
    std::string deltaFile = "deltaDataStream.bin";         // Delta data stream file path
    std::string outputFile = "output.wav";                 // Output file path (optional)
    
    // Vectors to hold left and right channels for Stream A and Stream B
    std::vector<double> A_left, A_right;
    std::vector<double> B_left, B_right;
    int fs_A, fs_B;
    
    // Read Stream A WAV file
    if (!readStereoWavFile(streamAFile, A_left, A_right, fs_A)) {
        return -1;
    }
    
    // Read Stream B WAV file
    if (!readStereoWavFile(streamBFile, B_left, B_right, fs_B)) {
        return -1;
    }
    
    // Ensure both files have the same sampling rate and length
    if (fs_A != fs_B) {
        std::cerr << "Sampling rates of Stream A and Stream B do not match." << std::endl;
        return -1;
    }
    
    size_t min_size = std::min({A_left.size(), A_right.size(), B_left.size(), B_right.size()});
    A_left.resize(min_size, 0.0);
    A_right.resize(min_size, 0.0);
    B_left.resize(min_size, 0.0);
    B_right.resize(min_size, 0.0);
    
    // Generate white noise carrier signals using NEON (if required)
    // Uncomment if you need to generate white noise carriers instead of using existing WAV files
    /*
    generateWhiteNoiseNEON(A_left, 0.7);
    generateWhiteNoiseNEON(A_right, 0.7);
    generateWhiteNoiseNEON(B_left, 0.7);
    generateWhiteNoiseNEON(B_right, 0.7);
    */
    
    // Parameters
    int frame_duration = 256;
    int number_of_features = 12;
    int Nfft = 512;
    std::vector<double> blackman_window = createBlackmanWindow(Nfft);
    std::vector<double> hamming_window = createHammingWindow(frame_duration);
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
        if (A_frame_left.size() < Nfft) {
            A_frame_left.resize(Nfft, 0.0);
            A_frame_right.resize(Nfft, 0.0);
            B_frame_left.resize(Nfft, 0.0);
            B_frame_right.resize(Nfft, 0.0);
        }
        
        // Compute FFT for Stream A Left
        std::vector<std::complex<double>> fft_A_L = computeFFT(A_frame_left, fft_plan, in_fft, out_fft);
        
        // Compute FFT for Stream A Right
        std::vector<std::complex<double>> fft_A_R = computeFFT(A_frame_right, fft_plan, in_fft, out_fft);
        
        // Compute FFT for Stream B Left
        std::vector<std::complex<double>> fft_B_L = computeFFT(B_frame_left, fft_plan, in_fft, out_fft);
        
        // Compute FFT for Stream B Right
        std::vector<std::complex<double>> fft_B_R = computeFFT(B_frame_right, fft_plan, in_fft, out_fft);
        
        // Inverse FFT and accumulate to Stream A and Stream B accumulators
        std::vector<double> ifft_A_L = computeIFFT(fft_A_L, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_A_R = computeIFFT(fft_A_R, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_B_L = computeIFFT(fft_B_L, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_B_R = computeIFFT(fft_B_R, ifft_plan, in_ifft, out_ifft);
        
        // Accumulate using NEON
        accumulateToOutputNEON(A_left_accum, ifft_A_L, begin_sample, frame_duration);
        accumulateToOutputNEON(A_right_accum, ifft_A_R, begin_sample, frame_duration);
        accumulateToOutputNEON(B_left_accum, ifft_B_L, begin_sample, frame_duration);
        accumulateToOutputNEON(B_right_accum, ifft_B_R, begin_sample, frame_duration);
        
        // Cepstral Analysis for Stream A Left
        std::vector<double> magnitude_spectrumA_L(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumA_L[n] = std::log(std::abs(fft_A_L[n]));
        }
        
        // Compute IFFT of log magnitude for Stream A Left
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumA_L[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepA_L = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepA_L(n) = out_ifft[n];
        }
        
        // Cepstral Analysis for Stream A Right
        std::vector<double> magnitude_spectrumA_R(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumA_R[n] = std::log(std::abs(fft_A_R[n]));
        }
        
        // Compute IFFT of log magnitude for Stream A Right
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumA_R[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepA_R = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepA_R(n) = out_ifft[n];
        }
        
        // Cepstral Analysis for Stream B Left
        std::vector<double> magnitude_spectrumB_L(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumB_L[n] = std::log(std::abs(fft_B_L[n]));
        }
        
        // Compute IFFT of log magnitude for Stream B Left
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumB_L[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepB_L = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepB_L(n) = out_ifft[n];
        }
        
        // Cepstral Analysis for Stream B Right
        std::vector<double> magnitude_spectrumB_R(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumB_R[n] = std::log(std::abs(fft_B_R[n]));
        }
        
        // Compute IFFT of log magnitude for Stream B Right
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumB_R[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepB_R = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepB_R(n) = out_ifft[n];
        }
        
        // Reconstruct the spectrum from cepstrum for Stream A Left
        Eigen::VectorXd cepA_padded_L = Eigen::VectorXd::Zero(Nfft);
        cepA_padded_L.head(number_of_features) = cepA_L;
        std::vector<double> cepA_padded_vec_L(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepA_padded_vec_L[n] = cepA_padded_L(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumA_L = computeFFT(cepA_padded_vec_L, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumA_L) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }
        
        // Reconstruct the spectrum from cepstrum for Stream A Right
        Eigen::VectorXd cepA_padded_R = Eigen::VectorXd::Zero(Nfft);
        cepA_padded_R.head(number_of_features) = cepA_R;
        std::vector<double> cepA_padded_vec_R(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepA_padded_vec_R[n] = cepA_padded_R(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumA_R = computeFFT(cepA_padded_vec_R, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumA_R) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }
        
        // Reconstruct the spectrum from cepstrum for Stream B Left
        Eigen::VectorXd cepB_padded_L = Eigen::VectorXd::Zero(Nfft);
        cepB_padded_L.head(number_of_features) = cepB_L;
        std::vector<double> cepB_padded_vec_L(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepB_padded_vec_L[n] = cepB_padded_L(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumB_L = computeFFT(cepB_padded_vec_L, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumB_L) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }
        
        // Reconstruct the spectrum from cepstrum for Stream B Right
        Eigen::VectorXd cepB_padded_R = Eigen::VectorXd::Zero(Nfft);
        cepB_padded_R.head(number_of_features) = cepB_R;
        std::vector<double> cepB_padded_vec_R(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepB_padded_vec_R[n] = cepB_padded_R(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumB_R = computeFFT(cepB_padded_vec_R, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumB_R) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }
        
        // Interpolate envelopes to Nfft
        std::vector<double> smoothA_L = linearInterpolate(std::vector<double>(cepA_padded_vec_L.begin(), cepA_padded_vec_L.begin() + number_of_features), Nfft);
        std::vector<double> smoothA_R = linearInterpolate(std::vector<double>(cepA_padded_vec_R.begin(), cepA_padded_vec_R.begin() + number_of_features), Nfft);
        std::vector<double> smoothB_L = linearInterpolate(std::vector<double>(cepB_padded_vec_L.begin(), cepB_padded_vec_L.begin() + number_of_features), Nfft);
        std::vector<double> smoothB_R = linearInterpolate(std::vector<double>(cepB_padded_vec_R.begin(), cepB_padded_vec_R.begin() + number_of_features), Nfft);
        
        // Flatten carrier spectra using NEON
        std::vector<std::complex<double>> flat_frameA_L = flattenCarrierSpectrumNEON(reconstructed_spectrumA_L, smoothA_L);
        std::vector<std::complex<double>> flat_frameA_R = flattenCarrierSpectrumNEON(reconstructed_spectrumA_R, smoothA_R);
        std::vector<std::complex<double>> flat_frameB_L = flattenCarrierSpectrumNEON(reconstructed_spectrumB_L, smoothB_L);
        std::vector<std::complex<double>> flat_frameB_R = flattenCarrierSpectrumNEON(reconstructed_spectrumB_R, smoothB_R);
        
        // Multiply with modulator's envelopes using NEON
        std::vector<std::complex<double>> XS_A_L = multiplyWithModEnvelopeNEON(flat_frameA_L, smoothA_L);
        std::vector<std::complex<double>> XS_A_R = multiplyWithModEnvelopeNEON(flat_frameA_R, smoothA_R);
        std::vector<std::complex<double>> XS_B_L = multiplyWithModEnvelopeNEON(flat_frameB_L, smoothB_L);
        std::vector<std::complex<double>> XS_B_R = multiplyWithModEnvelopeNEON(flat_frameB_R, smoothB_R);
        
        // Perform IFFT to get time-domain signals
        std::vector<double> ifft_XS_A_L = computeIFFT(XS_A_L, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_XS_A_R = computeIFFT(XS_A_R, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_XS_B_L = computeIFFT(XS_B_L, ifft_plan, in_ifft, out_ifft);
        std::vector<double> ifft_XS_B_R = computeIFFT(XS_B_R, ifft_plan, in_ifft, out_ifft);
        
        // Accumulate to xs_output using NEON
        accumulateToOutputNEON(xs_output, ifft_XS_A_L, begin_sample, frame_duration);
        accumulateToOutputNEON(xs_output, ifft_XS_A_R, begin_sample, frame_duration);
        accumulateToOutputNEON(xs_output, ifft_XS_B_L, begin_sample, frame_duration);
        accumulateToOutputNEON(xs_output, ifft_XS_B_R, begin_sample, frame_duration);
        
        // Create current frame's sparse matrices
        FrameSparseMatrices currentFrame(Nfft, Nfft);
        
        // Apply thresholding during insertion
        double threshold = 0.1; // Adjust based on your data
        
        // Stream A Left * Stream B Left
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(reconstructed_spectrumA_L[n] * reconstructed_spectrumB_L[n]);
            if (value > threshold) {
                currentFrame.A_L_B_L.insert(n, n) = value;
            }
        }
        
        // Stream A Left * Stream B Right
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(reconstructed_spectrumA_L[n] * reconstructed_spectrumB_R[n]);
            if (value > threshold) {
                currentFrame.A_L_B_R.insert(n, n) = value;
            }
        }
        
        // Stream A Right * Stream B Left
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(reconstructed_spectrumA_R[n] * reconstructed_spectrumB_L[n]);
            if (value > threshold) {
                currentFrame.A_R_B_L.insert(n, n) = value;
            }
        }
        
        // Stream A Right * Stream B Right
        for (int n = 0; n < Nfft; ++n) {
            double value = std::abs(reconstructed_spectrumA_R[n] * reconstructed_spectrumB_R[n]);
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
        
        // Store the delta frame
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
    serializeDeltaDataStream(deltaDataStream, deltaFile);
    std::cout << "Delta data stream serialized to " << deltaFile << std::endl;
    
    // Optionally, deserialize and reconstruct the data stream
    std::vector<FrameDeltaSparseMatrices> loadedDeltaDataStream = deserializeDeltaDataStream(deltaFile);
    std::cout << "Delta data stream deserialized from " << deltaFile << std::endl;
    
    std::vector<FrameSparseMatrices> reconstructedDataStream = reconstructDataStream(loadedDeltaDataStream, Nfft);
    std::cout << "Original data stream reconstructed from deltas." << std::endl;
    
    // Normalize xs_output
    double max_val = 0.0;
    for (const auto& sample : xs_output) {
        if (std::abs(sample) > max_val) {
            max_val = std::abs(sample);
        }
    }
    
    if (max_val > 1.0) {
        size_t i = 0;
        size_t size = xs_output.size();
        #pragma omp parallel for
        for (size_t i = 0; i + 1 < size; i += 2) {
            // Load two samples
            float64x2_t sample = vld1q_f64(&xs_output[i]);
            
            // Load max_val into NEON register
            float64x2_t max_vec = vdupq_n_f64(max_val);
            
            // Compute reciprocal of max_val
            float64x2_t recip_max = vrecpeq_f64(max_vec);
            recip_max = vmulq_f64(vrecpsq_f64(max_vec, recip_max), recip_max); // Refinement step
            
            // Perform division: sample * recip_max
            float64x2_t norm_sample = vmulq_f64(sample, recip_max);
            
            // Store back
            vst1q_f64(&xs_output[i], norm_sample);
        }
        
        // Handle remaining sample if size is odd
        for (; i < size; ++i) {
            xs_output[i] /= max_val;
        }
    }
    
    // Optionally, write the accumulated output to a WAV file
    // Uncomment the following lines if you wish to save the output audio
    /*
    if (!writeWavFile(outputFile, xs_output, fs_A)) {
        return -1;
    }
    std::cout << "Processing complete. Output saved to " << outputFile << std::endl;
    */
    
    // Instead of writing to a WAV file, you now have:
    // - dataStream: The full 4D data stream
    // - deltaDataStream: The delta-compressed 4D data stream
    // - reconstructedDataStream: Reconstructed data stream from deltas
    
    std::cout << "Processing complete. Delta compression applied to 4D sparse matrices." << std::endl;
    std::cout << "Total frames processed: " << dataStream.size() << std::endl;
    return 0;
}
