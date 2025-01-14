#ifndef WINDOWING_H
#define WINDOWING_H

#include <vector>

class Windowing {
public:
    static std::vector<double> createHammingWindow(int frame_duration);
    static void applyHammingWindowNEON(std::vector<double>& A_left, std::vector<double>& A_right,
                                      std::vector<double>& B_left, std::vector<double>& B_right,
                                      const std::vector<double>& hamming_window, size_t frame_duration);
};

#endif // WINDOWING_H