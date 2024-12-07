#ifndef WINDOWING_H
#define WINDOWING_H

#include <vector>

class Windowing {
public:
    static std::vector<double> createHammingWindow(int frame_duration);
    static void applyHammingWindow(std::vector<double>& frame, const std::vector<double>& hamming_window);
};

#endif // WINDOWING_H
