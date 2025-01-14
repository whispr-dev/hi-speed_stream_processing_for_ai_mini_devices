#ifndef AUDIOIO_H
#define AUDIOIO_H

#include <sndfile.h>
#include <vector>
#include <string>
#include <iostream>


class AudioIO {
public:
    AudioIO();
    bool readStereoWavFileValidated(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate);

private:
    // File paths (use std::string, not std::vector for file paths)
    static const std::string streamAFile;
    static const std::string streamBFile;
    static const std::string deltaFile;
    static const std::string featureFilePath;
    static const std::string hashMapFilePath;
};

#endif // AUDIOIO_H
