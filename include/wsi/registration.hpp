// Phase-correlation registration of two overlapping frames, and the small
// radix-2 FFT it runs on. Sizes must be powers of two; the caller crops the
// overlap strip to a power-of-two patch before calling.
#pragma once
#include <complex>
#include <vector>

#include "wsi/image.hpp"

namespace wsi {

void fft1d(std::vector<std::complex<float>>& a, bool inverse);
void fft2d(std::vector<std::complex<float>>& a, int w, int h, bool inverse);

struct Shift {
    // Where b was sampled relative to a, in pixels: if b comes from 7 px right
    // and 5 px above a, this is (+7, -5). Pinned by a signed unit test.
    float dx = 0.f, dy = 0.f;
    float peak = 0.f;  // correlation peak over the mean of the surface
    bool ok = false;
};

// Both images must be single-channel and the same power-of-two size.
Shift phase_correlate(const Image& a, const Image& b);

}  // namespace wsi
