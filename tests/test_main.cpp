// Unit tests. No framework: a CHECK macro and a main that returns non-zero.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "wsi/camera.hpp"
#include "wsi/classifier.hpp"
#include "wsi/focus.hpp"
#include "wsi/registration.hpp"
#include "wsi/rng.hpp"
#include "wsi/scan.hpp"
#include "wsi/segment.hpp"
#include "wsi/slide.hpp"
#include "wsi/tiff.hpp"

using namespace wsi;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                              \
    do {                                                              \
        ++g_checks;                                                   \
        if (!(cond)) {                                                \
            ++g_failures;                                             \
            std::printf("  FAIL %s:%d  ", __FILE__, __LINE__);        \
            std::printf(__VA_ARGS__);                                 \
            std::printf("\n");                                        \
        }                                                             \
    } while (0)

static Image test_texture(int w, int h, uint64_t seed) {
    Image img(w, h, 1);
    Rng r(seed);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img.at(x, y) = uint8_t(128 + 60 * std::sin(x * 0.31 + y * 0.17) +
                                   30 * r.normal() * 0.3);
    return img;
}

static void test_otsu() {
    std::printf("otsu\n");
    uint64_t hist[256] = {0};
    for (int i = 40; i < 60; ++i) hist[i] = 1000;    // dark mode  (tissue)
    for (int i = 200; i < 220; ++i) hist[i] = 3000;  // bright mode (glass)
    int t = otsu_threshold(hist);
    // Every t in [59, 199] separates the modes equally well; the implementation
    // returns the first maximiser.
    CHECK(t >= 59 && t < 200, "threshold %d not between the two modes", t);

    // A flat histogram must still return something usable, not a crash.
    uint64_t flat[256];
    for (int i = 0; i < 256; ++i) flat[i] = 10;
    int tf = otsu_threshold(flat);
    CHECK(tf >= 0 && tf <= 255, "flat histogram threshold %d out of range", tf);
}

static void test_fft_roundtrip() {
    std::printf("fft roundtrip\n");
    const int n = 64;
    std::vector<std::complex<float>> a(n), orig;
    Rng r(9);
    for (int i = 0; i < n; ++i) a[size_t(i)] = std::complex<float>(float(r.uniform(-1, 1)), 0.f);
    orig = a;
    fft1d(a, false);
    fft1d(a, true);
    double worst = 0;
    for (int i = 0; i < n; ++i) worst = std::max(worst, double(std::abs(a[size_t(i)] - orig[size_t(i)])));
    CHECK(worst < 1e-4, "fft roundtrip error %g", worst);
}

static void test_phase_correlation() {
    std::printf("phase correlation\n");
    Image big = test_texture(256, 256, 3);
    const int shift_x = 7, shift_y = -5;
    Image a = crop(big, 64, 64, 128, 128);
    Image b = crop(big, 64 + shift_x, 64 + shift_y, 128, 128);
    Shift s = phase_correlate(a, b);
    std::printf("  [convention] b sampled 7 right / 5 up of a -> dx %.2f dy %.2f peak %.1f\n", s.dx,
                s.dy, s.peak);
    CHECK(s.ok, "phase correlation refused the input");
    // Signed, not just in magnitude: the mosaic places tiles with these numbers,
    // and a sign error there is a silently mirrored slide.
    CHECK(std::abs(s.dx - double(shift_x)) < 0.5, "dx %.2f, expected %d", s.dx, shift_x);
    CHECK(std::abs(s.dy - double(shift_y)) < 0.5, "dy %.2f, expected %d", s.dy, shift_y);

    Image odd(100, 128, 1);
    CHECK(!phase_correlate(odd, odd).ok, "non power-of-two input should be rejected");
}

static void test_focus_metric_monotonic() {
    std::printf("focus metrics fall off with blur\n");
    Image sharp = test_texture(128, 128, 11);
    double prev_b = brenner(sharp), prev_v = var_laplacian(sharp);
    for (float sigma : {1.0f, 2.0f, 4.0f}) {
        Image blurred = sharp;
        blur_gaussian(blurred, sigma);
        double b = brenner(blurred), v = var_laplacian(blurred);
        CHECK(b < prev_b, "brenner did not drop at sigma %.1f (%.1f -> %.1f)", sigma, prev_b, b);
        CHECK(v < prev_v, "var-laplacian did not drop at sigma %.1f", sigma);
        prev_b = b;
        prev_v = v;
    }
}

static void test_tiff_roundtrip() {
    std::printf("pyramidal tiff roundtrip\n");
    Image src(700, 500, 3);
    Rng r(21);
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < src.w; ++x)
            for (int c = 0; c < 3; ++c) src.at(x, y, c) = uint8_t((x * 3 + y * 5 + c * 40) & 0xFF);

    const std::string path = "test_pyramid.tif";
    PyramidStats st = write_pyramidal_tiff(path, src, 256);
    CHECK(st.levels.size() >= 3, "expected >= 3 levels, got %zu", st.levels.size());
    CHECK(st.levels[0].w == 700 && st.levels[0].h == 500, "level 0 dims wrong");
    CHECK(st.levels[1].w == 350 && st.levels[1].h == 250, "level 1 dims wrong");

    TiffInfo info = read_tiff_info(path);
    CHECK(info.ok, "could not parse back the TIFF we just wrote");
    CHECK(info.levels.size() == st.levels.size(), "level count mismatch on read-back");
    CHECK(info.tile_size == 256, "tile size %d on read-back", info.tile_size);
    CHECK(info.channels == 3, "channels %d on read-back", info.channels);

    Image tile = read_tiff_tile(path, 0, 1, 1);
    CHECK(tile.w == 256 && tile.h == 256, "tile dims wrong");
    bool match = true;
    for (int y = 0; y < 64 && match; ++y)
        for (int x = 0; x < 64; ++x)
            if (tile.at(x, y, 0) != src.at(256 + x, 256 + y, 0)) { match = false; break; }
    CHECK(match, "tile (1,1) pixels do not match the source image");

    // The bottom-right tile is padded; the padding must be glass white.
    const TiffLevel& l0 = st.levels[0];
    Image edge = read_tiff_tile(path, 0, l0.tiles_x - 1, l0.tiles_y - 1);
    CHECK(edge.at(255, 255, 0) == 255, "edge tile padding is %d, expected 255", edge.at(255, 255, 0));
    std::remove(path.c_str());
}

static void test_flatfield() {
    std::printf("flat-field correction\n");
    OpticsParams p;
    p.fov_w = p.fov_h = 128;
    p.noise_sigma = 0.0;
    Image slide(256, 256, 3);
    SyntheticStage stage;
    SyntheticCamera cam(slide, stage, p);

    Image blank = cam.blank_frame();
    double centre = blank.at(64, 64, 0), corner = blank.at(2, 2, 0);
    CHECK(corner < centre * 0.90, "vignette too weak to test: corner %.0f centre %.0f", corner, centre);

    FlatField ff = estimate_flatfield(blank);
    Image corrected = blank;
    apply_flatfield(corrected, ff);
    double ratio = double(corrected.at(2, 2, 0)) / double(corrected.at(64, 64, 0));
    CHECK(std::abs(ratio - 1.0) < 0.05, "corner/centre after correction %.3f", ratio);
}

static void test_planner_serpentine() {
    std::printf("serpentine planner\n");
    TissueSeg seg;
    seg.mask = TissueMask{16, 16, 64, std::vector<uint8_t>(256, 255)};
    seg.x0 = 0; seg.y0 = 0; seg.x1 = 1024; seg.y1 = 1024;
    ScanPlan plan = plan_serpentine(seg, 1024, 1024, 256, 256, 64);
    CHECK(plan.cols > 1 && plan.rows > 1, "grid too small: %dx%d", plan.cols, plan.rows);
    CHECK(plan.step_x == 192, "step_x %d", plan.step_x);

    // Row 0 runs left to right, row 1 runs right to left.
    CHECK(plan.tiles[0].col == 0, "row 0 should start at col 0");
    int first_row1 = -1;
    for (size_t i = 0; i < plan.tiles.size(); ++i)
        if (plan.tiles[i].row == 1) { first_row1 = plan.tiles[i].col; break; }
    CHECK(first_row1 == plan.cols - 1, "row 1 should start at the far column, got %d", first_row1);

    // With no tissue at all, nothing is queued for acquisition.
    TissueSeg empty;
    empty.mask = TissueMask{16, 16, 64, std::vector<uint8_t>(256, 0)};
    empty.x1 = 1024; empty.y1 = 1024;
    ScanPlan none = plan_serpentine(empty, 1024, 1024, 256, 256, 64);
    CHECK(none.to_acquire == 0, "%d tiles queued on an empty slide", none.to_acquire);
}

static void test_autofocus_finds_truth() {
    std::printf("autofocus finds the true focal plane\n");
    SlideSpec spec;
    spec.w = 1536;
    spec.h = 1024;
    spec.blobs = 2;
    spec.folds = 0;
    spec.bubbles = 0;
    Image slide = generate_slide(spec);

    OpticsParams p;
    p.fov_w = p.fov_h = 256;
    SyntheticStage stage;
    SyntheticCamera cam(slide, stage, p);

    AfConfig cfg;
    cfg.roi = 128;
    stage.move_xy(600, 400);
    AfResult r = autofocus_sweep(stage, cam, cfg);
    double truth = cam.true_focus_z(600, 400);
    CHECK(std::abs(r.z - truth) < 0.8, "autofocus %.2f um vs truth %.2f um", r.z, truth);
    CHECK(r.frames > 4, "suspiciously few frames: %d", r.frames);
}

static void test_downsample_fast_path_matches() {
    std::printf("2x downsample fast path is bit-identical\n");
    const std::pair<int, int> sizes[] = {{64, 48}, {65, 49}, {257, 129}};
    for (const auto& dims : sizes) {
        for (int ch : {1, 3}) {
            Image src(dims.first, dims.second, ch);
            Rng r(uint64_t(dims.first * 31 + ch));
            for (auto& p : src.px) p = uint8_t(r.next() & 0xFF);
            CHECK(downsample2(src).px == downsample(src, 2).px,
                  "fast 2x differs from the generic path at %dx%d ch=%d", dims.first, dims.second,
                  ch);
        }
    }
}

static void test_prepare_tile() {
    std::printf("tile preparation for the classifier\n");
    Image src(300, 200, 3);
    for (int y = 0; y < src.h; ++y)
        for (int x = 0; x < src.w; ++x)
            for (int c = 0; c < 3; ++c) src.at(x, y, c) = uint8_t((x + y) & 0xFF);
    Image t = prepare_tile(src, 64);
    CHECK(t.w == 64 && t.h == 64 && t.ch == 1, "prepared tile is %dx%dx%d", t.w, t.h, t.ch);
}

int main() {
    test_otsu();
    test_fft_roundtrip();
    test_phase_correlation();
    test_focus_metric_monotonic();
    test_tiff_roundtrip();
    test_flatfield();
    test_planner_serpentine();
    test_autofocus_finds_truth();
    test_downsample_fast_path_matches();
    test_prepare_tile();

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
