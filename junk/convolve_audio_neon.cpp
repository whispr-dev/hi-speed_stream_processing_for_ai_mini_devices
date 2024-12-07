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

// Function to read WAV file
bool readWavFile(const std::string& filename, std::vector<double>& data, int& sampleRate) {
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

    // If stereo, convert to mono by averaging channels
    if (numChannels == 1) {
        data = tempData;
    } else {
        data.resize(numFrames);
        for (int i = 0; i < numFrames; ++i) {
            double sum = 0.0;
            for (int ch = 0; ch < numChannels; ++ch) {
                sum += tempData[i * numChannels + ch];
            }
            data[i] = sum / numChannels;
        }
    }

    return true;
}

// Function to write WAV file
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

int main() {
    std::string inputFile = "../data/dropitonmeMono.wav"; // Replace with your input file path
    std::string outputFile = "output.wav"; // Output file path
    std::vector<double> y; // Modulator
    std::vector<double> x; // Carrier
    int fs;

    // Read WAV file
    if (!readWavFile(inputFile, y, fs)) {
        return -1;
    }

    // Initialize carrier with white noise using NEON
    x = y; // Same size
    generateWhiteNoiseNEON(x, 0.7);

    // Parameters
    int frame_duration = 256;
    int number_of_features = 12;
    int Nfft = 512;
    std::vector<double> blackman_window = createBlackmanWindow(Nfft);
    std::vector<double> hamming_window = createHammingWindow(frame_duration);
    int hopSize = frame_duration / 4; // 75% overlap
    int number_of_frames = (y.size() - frame_duration) / hopSize + 1;

    // Initialize accumulators
    std::vector<double> mod(y.size(), 0.0);
    std::vector<double> car(x.size(), 0.0);
    std::vector<double> xs_output(y.size(), 0.0);

    // Placeholder for cepstrum matrices
    std::vector<Eigen::VectorXd> dftcepstraM(number_of_frames, Eigen::VectorXd(number_of_features));
    std::vector<Eigen::VectorXd> dftcepstraC(number_of_frames, Eigen::VectorXd(number_of_features));

    // Initialize FFTW plans
    fftw_complex* in_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_complex* out_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    fftw_plan fft_plan = fftw_plan_dft_1d(Nfft, in_fft, out_fft, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_complex* in_ifft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Nfft);
    double* out_ifft = (double*)fftw_malloc(sizeof(double) * Nfft);
    fftw_plan ifft_plan = fftw_plan_dft_c2r_1d(Nfft, in_ifft, out_ifft, FFTW_ESTIMATE);

    // Processing loop
    for (int i = 0; i < number_of_frames; ++i) {
        int begin_sample = i * hopSize;
        int end_sample = begin_sample + frame_duration;

        // Ensure we don't go out of bounds
        if (end_sample > y.size()) {
            end_sample = y.size();
        }

        // Extract frames
        std::vector<double> s(y.begin() + begin_sample, y.begin() + end_sample);
        std::vector<double> c(x.begin() + begin_sample, x.begin() + end_sample);

        // Apply Hamming window using NEON
        applyHammingWindowNEON(s, hamming_window);
        applyHammingWindowNEON(c, hamming_window);

        // Pad frames to Nfft with zeros if necessary
        if (s.size() < Nfft) {
            s.resize(Nfft, 0.0);
        }
        if (c.size() < Nfft) {
            c.resize(Nfft, 0.0);
        }

        // Compute FFT of modulator frame
        std::vector<std::complex<double>> frame_fftM = computeFFT(s, fft_plan, in_fft, out_fft);

        // Inverse FFT and accumulate to mod
        std::vector<double> ifft_resultM = computeIFFT(frame_fftM, ifft_plan, in_ifft, out_ifft);
        // Accumulate using NEON
        accumulateToOutputNEON(mod, ifft_resultM, begin_sample, frame_duration);

        // Compute FFT of carrier frame
        std::vector<std::complex<double>> frame_fftC = computeFFT(c, fft_plan, in_fft, out_fft);

        // Inverse FFT and accumulate to car
        std::vector<double> ifft_resultC = computeIFFT(frame_fftC, ifft_plan, in_ifft, out_ifft);
        // Accumulate using NEON
        accumulateToOutputNEON(car, ifft_resultC, begin_sample, frame_duration);

        // Cepstra calculation for modulator
        std::vector<double> magnitude_spectrumM(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumM[n] = std::log(std::abs(frame_fftM[n]));
        }

        // Compute IFFT of log magnitude for modulator
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumM[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepM = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepM(n) = out_ifft[n];
        }
        dftcepstraM[i] = cepM;

        // Cepstra calculation for carrier
        std::vector<double> magnitude_spectrumC(frame_duration, 0.0);
        for (int n = 0; n < frame_duration; ++n) {
            magnitude_spectrumC[n] = std::log(std::abs(frame_fftC[n]));
        }

        // Compute IFFT of log magnitude for carrier
        for (int n = 0; n < frame_duration; ++n) {
            in_ifft[n][0] = magnitude_spectrumC[n];
            in_ifft[n][1] = 0.0;
        }
        fftw_execute(ifft_plan);
        Eigen::VectorXd cepC = Eigen::VectorXd::Zero(number_of_features);
        for (int n = 0; n < number_of_features; ++n) {
            cepC(n) = out_ifft[n];
        }
        dftcepstraC[i] = cepC;

        // Reconstruct the spectrum from cepstrum for modulator
        Eigen::VectorXd cepM_padded = Eigen::VectorXd::Zero(Nfft);
        cepM_padded.head(number_of_features) = cepM;
        std::vector<double> cepM_padded_vec(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepM_padded_vec[n] = cepM_padded(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumM = computeFFT(cepM_padded_vec, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumM) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }

        // Reconstruct the spectrum from cepstrum for carrier
        Eigen::VectorXd cepC_padded = Eigen::VectorXd::Zero(Nfft);
        cepC_padded.head(number_of_features) = cepC;
        std::vector<double> cepC_padded_vec(Nfft, 0.0);
        for (int n = 0; n < number_of_features; ++n) {
            cepC_padded_vec[n] = cepC_padded(n);
        }
        std::vector<std::complex<double>> reconstructed_spectrumC = computeFFT(cepC_padded_vec, fft_plan, in_fft, out_fft);
        // Exponentiate to get magnitude spectrum
        for (auto& val : reconstructed_spectrumC) {
            val = std::exp(val.real()) + std::complex<double>(0.0, 0.0);
        }

        // Interpolate envelopes to Nfft
        std::vector<double> smoothM = linearInterpolate(std::vector<double>(cepM_padded_vec.begin(), cepM_padded_vec.begin() + number_of_features), Nfft);
        std::vector<double> smoothC = linearInterpolate(std::vector<double>(cepC_padded_vec.begin(), cepC_padded_vec.begin() + number_of_features), Nfft);

        // Flatten carrier spectrum using NEON
        std::vector<std::complex<double>> flat_frameC = flattenCarrierSpectrumNEON(reconstructed_spectrumC, smoothC);

        // Multiply with modulator's envelope using NEON
        std::vector<std::complex<double>> XS = multiplyWithModEnvelopeNEON(flat_frameC, smoothM);

        // Perform IFFT to get time-domain signal
        std::vector<double> ifft_resultXS = computeIFFT(XS, ifft_plan, in_ifft, out_ifft);

        // Accumulate to xs_output using NEON
        accumulateToOutputNEON(xs_output, ifft_resultXS, begin_sample, frame_duration);
    }

    // Clean up FFTW resources
    fftw_destroy_plan(fft_plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in_fft);
    fftw_free(out_fft);
    fftw_free(in_ifft);
    fftw_free(out_ifft);

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
        for (; i + 1 < size; i += 2) {
            // Load two samples
            float64x2_t sample = vld1q_f64(&xs_output[i]);
            
            // Load max_val into NEON register
            float64x2_t max_vec = vdupq_n_f64(max_val);
            
            // Perform division
            float64x2_t norm_sample = vmulq_f64(sample, vrecpeq_f64(max_vec));
            norm_sample = vmulq_f64(vrecpsq_f64(max_vec, norm_sample), norm_sample);
            
            // Store back
            vst1q_f64(&xs_output[i], norm_sample);
        }

        // Handle remaining sample if size is odd
        for (; i < size; ++i) {
            xs_output[i] /= max_val;
        }
    }

    // Write output WAV file
    if (!writeWavFile(outputFile, xs_output, fs)) {
        return -1;
    }

    std::cout << "Processing complete. Output saved to " << outputFile << std::endl;
    return 0;
}
