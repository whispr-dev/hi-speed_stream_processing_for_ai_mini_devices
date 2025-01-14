#ifndef AUDIOIO_H
#define AUDIOIO_H

#include <sndfile.h>
#include <vector>
#include <string>

class AudioIO {
public:
    AudioIO();
    bool readStereoWavFileValidated(const std::string& filename, std::vector<double>& left, std::vector<double>& right, int& sampleRate);

private:
    static const std::string streamAFile;
    static const std::string streamBFile;
};

#endif // AUDIOIO_H
