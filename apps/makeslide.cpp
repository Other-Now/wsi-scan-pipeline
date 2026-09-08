// Generates the source slide the synthetic camera scans.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "wsi/slide.hpp"

using namespace wsi;

static void usage() {
    std::printf(
        "usage: wsi_makeslide [--out slide.ppm] [--width 6144] [--height 4096]\n"
        "                     [--seed 42] [--blobs 5] [--folds 3] [--bubbles 4]\n"
        "                     [--truth slide_truth.json]\n");
}

int main(int argc, char** argv) {
    SlideSpec spec;
    std::string out = "slide.ppm";
    std::string truth_path = "slide_truth.json";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--out") out = next();
        else if (a == "--width") spec.w = std::atoi(next().c_str());
        else if (a == "--height") spec.h = std::atoi(next().c_str());
        else if (a == "--seed") spec.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--blobs") spec.blobs = std::atoi(next().c_str());
        else if (a == "--folds") spec.folds = std::atoi(next().c_str());
        else if (a == "--bubbles") spec.bubbles = std::atoi(next().c_str());
        else if (a == "--truth") truth_path = next();
        else { usage(); return a == "--help" ? 0 : 2; }
    }

    auto t0 = std::chrono::steady_clock::now();
    SlideTruth truth;
    Image slide = generate_slide(spec, &truth);
    auto t1 = std::chrono::steady_clock::now();
    write_pnm(out, slide);
    write_truth_json(truth_path, spec, truth);
    auto t2 = std::chrono::steady_clock::now();

    std::printf("slide     %d x %d, %.1f MB\n", slide.w, slide.h, double(slide.bytes()) / 1e6);
    std::printf("artifacts %zu (%d folds, %d bubbles)\n", truth.artifacts.size(), spec.folds,
                spec.bubbles);
    std::printf("generate  %.0f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    std::printf("write     %.0f ms -> %s\n",
                std::chrono::duration<double, std::milli>(t2 - t1).count(), out.c_str());
    return 0;
}
