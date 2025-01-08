int main() {
    // File paths for two stereo WAV files (Stream A and Stream B)
    std::string streamAFile = "../data/streamA_stereo.wav"; // Replace with your Stream A file path
    std::string streamBFile = "../data/streamB_stereo.wav"; // Replace with your Stream B file path
    std::string deltaFile = "deltaDataStream.bin";         // Delta data stream file path
    std::string featureFilePath = "featureMatrix.csv";     // Feature matrix file path (optional)
    std::string hashMapFilePath = "frameHashMap.txt";      // Hash map file path (optional)
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
    int number_of_features = 12; // Adjust based on your feature extraction
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
    // Function to serialize a sparse matrix and FrameDeltaSparseMatrices are defined above
    // Function to buildFrameHashMap and buildFeatureMatrix are defined above
    serializeDeltaDataStream(deltaDataStream, deltaFile);
    std::cout << "Delta data stream serialized to " << deltaFile << std::endl;
    
    // ------------------------------
    // Parsing and Preparation Module
    // ------------------------------
    
    // Step 1: Build the hash map from the delta data stream
    FrameHashMap frameMap = buildFrameHashMap(deltaDataStream);
    
    // Step 2: Extract features and build the feature matrix for ML
    Eigen::MatrixXd featureMatrix = buildFeatureMatrix(deltaDataStream, frameMap, number_of_features);
    
    // Step 3: Save the feature matrix to a file for ML training (optional)
    std::ofstream featureFile(featureFilePath);
    if (featureFile.is_open()) {
        featureFile << featureMatrix.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
        featureFile.close();
        std::cout << "Feature matrix saved to " << featureFilePath << std::endl;
    } else {
        std::cerr << "Unable to open file for writing feature matrix." << std::endl;
    }
    
    // Step 4: Save the hash map to a file for reference (optional)
    std::ofstream hashMapFile(hashMapFilePath);
    if (hashMapFile.is_open()) {
        for (const auto& pair : frameMap) {
            hashMapFile << pair.second << ": " << pair.first << "\n";
        }
        hashMapFile.close();
        std::cout << "Hash map saved to " << hashMapFilePath << std::endl;
    } else {
        std::cerr << "Unable to open file for writing hash map." << std::endl;
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
    if (!writeWavFile(outputFile, xs_output, fs_A)) {
        return -1;
    }
    std::cout << "Processing complete. Output saved to " << outputFile << std::endl;
    */
    
    // Instead of writing to a WAV file, you now have:
    // - dataStream: The full 4D data stream
    // - deltaDataStream: The delta-compressed 4D data stream
    // - frameMap: Mapping of unique frames to unique IDs
    // - featureMatrix: Matrix of features ready for ML
    
    std::cout << "Processing complete. Delta compression and parsing/preparation modules executed successfully." << std::endl;
    std::cout << "Total frames processed: " << dataStream.size() << std::endl;
    
    return 0;
}
