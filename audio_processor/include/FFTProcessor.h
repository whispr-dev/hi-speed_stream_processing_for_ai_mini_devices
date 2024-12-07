#ifndef FFTPROCESSOR_H
#define FFTPROCESSOR_H

#include <complex>
#include <vector>

class FFTProcessor {
public:
    FFTProcessor(int Nfft);
    ~FFTProcessor();

    std::vector<std::complex<double>> computeFFT(const std::vector<double>& frame);
    std::vector<double> computeIFFT(const std::vector<std::complex<double>>& spectrum);

private:
    int Nfft_;
    // FFTW plans and other necessary members
};

#endif // FFTPROCESSOR_H
