#include <gtest/gtest.h>
#include "FeatureExtractor.h"

TEST(FeatureExtractorTest, ComputeLogMelEnergies) {
    FeatureExtractor fe(/* parameters */);
    std::vector<float> mel_energies = {1.0f, 2.0f, 4.0f, 8.0f};
    fe.compute_log_mel_energies_neon(mel_energies.data(), mel_energies.size());
    
    // Expected results
    std::vector<float> expected = {0.0f, 0.693147f, 1.386294f, 2.079442f};
    
    for (size_t i = 0; i < mel_energies.size(); ++i) {
        EXPECT_NEAR(mel_energies[i], expected[i], 1e-3);
    }
}
