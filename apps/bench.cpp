// Throughput of the three inner loops the scanner spends its time in.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "wsi/focus.hpp"
#include "wsi/image.hpp"
#include "wsi/registration.hpp"
#include "wsi/rng.hpp"

using namespace wsi;
using Clock = std::chrono::steady_clock;

namespace {

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Image noisy(int w, int h, int ch, uint64_t seed) {
    Image img(w, h, ch);
    Rng r(seed);
    for (size_t i = 0; i < img.px.size(); ++i) img.px[i] = uint8_t(r.next() & 0xFF);
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    if (reps < 1) reps = 1;

    Image big = noisy(4096, 4096, 3, 1);
    const double src_bytes = double(big.bytes());

    // 2x box downsample: the pyramid build, and the only place where the layout
    // of the image buffer shows up in the wall clock.
    {
        double best_generic = 1e30, best_fast = 1e30;
        for (int i = 0; i < reps; ++i) {
            auto t0 = Clock::now();
            Image a = downsample(big, 2);
            auto t1 = Clock::now();
            Image b = downsample2(big);
            auto t2 = Clock::now();
            best_generic = std::min(best_generic, ms(t0, t1));
            best_fast = std::min(best_fast, ms(t1, t2));
            if (a.px != b.px) {
                std::printf("MISMATCH: the fast path is not bit-identical\n");
                return 1;
            }
        }
        double touched = src_bytes * 1.25;  // read the source, write a quarter of it
        std::printf("downsample 2x   4096x4096x3\n");
        std::printf("  generic NxN   %7.1f ms  %5.2f GB/s\n", best_generic,
                    touched / (best_generic * 1e-3) / 1e9);
        std::printf("  specialised   %7.1f ms  %5.2f GB/s   (%.2fx)\n", best_fast,
                    touched / (best_fast * 1e-3) / 1e9, best_generic / best_fast);
    }

    // Focus metrics: run once per autofocus frame, on the AF ROI.
    {
        Image roi = to_gray(noisy(192, 192, 3, 2));
        const int n = 200;
        auto t0 = Clock::now();
        double sink = 0;
        for (int i = 0; i < n; ++i) sink += brenner(roi);
        auto t1 = Clock::now();
        for (int i = 0; i < n; ++i) sink += var_laplacian(roi);
        auto t2 = Clock::now();
        double mp = double(roi.w) * roi.h * n / 1e6;
        std::printf("focus metrics   192x192 ROI, %d iterations (sink %.0f)\n", n, sink * 0.0);
        std::printf("  brenner       %7.3f ms/frame  %6.1f MPix/s\n", ms(t0, t1) / n,
                    mp / (ms(t0, t1) * 1e-3));
        std::printf("  var-laplacian %7.3f ms/frame  %6.1f MPix/s\n", ms(t1, t2) / n,
                    mp / (ms(t1, t2) * 1e-3));
    }

    // Phase correlation: two FFTs, a spectrum multiply, and one inverse FFT per
    // overlapping pair of FOVs.
    for (int patch : {64, 128, 256}) {
        Image a = to_gray(noisy(patch, patch, 3, 3));
        Image b = to_gray(noisy(patch, patch, 3, 4));
        const int n = patch <= 128 ? 200 : 50;
        auto t0 = Clock::now();
        for (int i = 0; i < n; ++i) (void)phase_correlate(a, b);
        auto t1 = Clock::now();
        std::printf("phase corr      %3dx%-3d       %7.3f ms/pair\n", patch, patch, ms(t0, t1) / n);
    }
    return 0;
}
