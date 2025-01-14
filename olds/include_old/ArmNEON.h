#ifdef __ARM_NEON
// ARM SIMD code using NEON intrinsics
#include <arm_neon.h>
#elif defined(__SSE__)
// x86 SIMD code using SSE/AVX intrinsics
#include <xmmintrin.h>
#endif
