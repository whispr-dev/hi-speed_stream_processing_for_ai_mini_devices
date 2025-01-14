cmake_minimum_required(VERSION 3.12)
project(audio_processor)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable (target)
add_executable(audio_processor
    src/main.cpp
    src/AudioIO.cpp
    src/DeltaProcessor.cpp
    src/Utils.cpp
    src/FeatureExtractor.cpp
    src/FFTProcessor.cpp
    src/Windowing.cpp
)

target_include_directories(audio_processor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    C:/Users/phine/Desktop/audio_processor/lib/libsndfile-1.2.2-win64/bin
)

target_link_libraries(audio_processor PRIVATE
lib/libsndfile-1.2.2-win64/include/sndfile
)


# Include directories
target_include_directories(audio_processor PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    C:/Users/phine/Desktop/audio_processor/lib/libsndfile-1.2.2-win64/include
)

# Link libraries
target_link_libraries(audio_processor PRIVATE lib/libsndfile-1.2.2-win64/include/sndfile
)
