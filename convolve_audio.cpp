#include <sndfile.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>

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
    for (size_t i = 0; i < data.size(); ++i) {
        double sample = data[i];
        // Clipping
        if (sample > 1.0) sample = 1.0;
        if (sample < -1.0) sample = -1.0;
        intData[i] = static_cast<short>(sample * 32767);
    }

    sf_writef_short(file, intData.data(), data.size());
    sf_close(file);

    return true;
}

// Function to generate white noise
void generateWhiteNoise(std::vector<double>& carrier, double amplitude = 0.7) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (auto& sample : carrier) {
        sample = amplitude * dis(gen);
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

// Function to flatten carrier spectrum by dividing by its envelope
std::vector<std::complex<double>> flattenCarrierSpectrum(const std::vector<std::complex<double>>& carrierSpectrum, const std::vector<double>& envelope) {
    std::vector<std::complex<double>> flattenedSpectrum(carrierSpectrum.size());
    for (size_t n = 0; n < carrierSpectrum.size(); ++n) {
        if (envelope[n] != 0.0) {
            flattenedSpectrum[n] = carrierSpectrum[n] / envelope[n];
        } else {
            flattenedSpectrum[n] = std::complex<double>(0.0, 0.0);
        }
    }
    return flattenedSpectrum;
}

// Function to multiply with modulator's envelope
std::vector<std::complex<double>> multiplyWithModEnvelope(const std::vector<std::complex<double>>& spectrum, const std::vector<double>& modEnvelope) {
    std::vector<std::complex<double>> multipliedSpectrum(spectrum.size());
    for (size_t n = 0; n < spectrum.size(); ++n) {
        multipliedSpectrum[n] = spectrum[n] * modEnvelope[n];
    }
    return multipliedSpectrum;
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

    // Initialize carrier with white noise
    x = y; // Same size
    generateWhiteNoise(x, 0.7);

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

        // Apply Hamming window
        for (int n = 0; n < frame_duration; ++n) {
            if (n < s.size()) {
                s[n] *= hamming_window[n];
            }
            if (n < c.size()) {
                c[n] *= hamming_window[n];
            }
        }

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
        for (int n = 0; n < frame_duration; ++n) {
            if (begin_sample + n < mod.size()) {
                mod[begin_sample + n] += ifft_resultM[n];
            }
        }

        // Compute FFT of carrier frame
        std::vector<std::complex<double>> frame_fftC = computeFFT(c, fft_plan, in_fft, out_fft);

        // Inverse FFT and accumulate to car
        std::vector<double> ifft_resultC = computeIFFT(frame_fftC, ifft_plan, in_ifft, out_ifft);
        for (int n = 0; n < frame_duration; ++n) {
            if (begin_sample + n < car.size()) {
                car[begin_sample + n] += ifft_resultC[n];
            }
        }

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

        // Flatten carrier spectrum
        std::vector<std::complex<double>> flat_frameC = flattenCarrierSpectrum(reconstructed_spectrumC, smoothC);

        // Multiply with modulator's envelope
        std::vector<std::complex<double>> XS = multiplyWithModEnvelope(flat_frameC, smoothM);

        // Perform IFFT to get time-domain signal
        std::vector<double> ifft_resultXS = computeIFFT(XS, ifft_plan, in_ifft, out_ifft);

        // Accumulate to xs_output
        for (int n = 0; n < frame_duration; ++n) {
            if (begin_sample + n < xs_output.size()) {
                xs_output[begin_sample + n] += ifft_resultXS[n];
            }
        }
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
        for (auto& sample : xs_output) {
            sample /= max_val;
        }
    }

    // Write output WAV file
    if (!writeWavFile(outputFile, xs_output, fs)) {
        return -1;
    }

    std::cout << "Processing complete. Output saved to " << outputFile << std::endl;
    return 0;
}
