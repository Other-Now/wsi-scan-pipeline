#include "wsi/slide.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "wsi/rng.hpp"

namespace wsi {
namespace {

// Tissue outline as a sum of gaussian blobs, evaluated on a coarse grid and
// bilinearly upsampled. Evaluating it per pixel would dominate the runtime and
// would not look any different.
struct DensityField {
    int gw = 0, gh = 0, step = 8;
    std::vector<float> v;
    float sample(double x, double y) const {
        double gx = std::clamp(x / step, 0.0, double(gw - 1));
        double gy = std::clamp(y / step, 0.0, double(gh - 1));
        int x0 = int(gx), y0 = int(gy);
        int x1 = std::min(x0 + 1, gw - 1), y1 = std::min(y0 + 1, gh - 1);
        double fx = gx - x0, fy = gy - y0;
        double a = v[size_t(y0) * gw + x0] * (1 - fx) + v[size_t(y0) * gw + x1] * fx;
        double b = v[size_t(y1) * gw + x0] * (1 - fx) + v[size_t(y1) * gw + x1] * fx;
        return float(a * (1 - fy) + b * fy);
    }
};

struct Blob {
    double cx, cy, rx, ry, rot;
};

}  // namespace

Image generate_slide(const SlideSpec& spec, SlideTruth* truth) {
    Rng rng(spec.seed);
    Image img(spec.w, spec.h, 3);

    std::vector<Blob> blobs;
    for (int i = 0; i < spec.blobs; ++i) {
        // Sized so the sections cover roughly a fifth of the glass, which is
        // what a real slide looks like and what makes skipping background worth
        // measuring. Big overlapping blobs would flatter the planner.
        blobs.push_back({rng.uniform(0.15, 0.85) * spec.w, rng.uniform(0.15, 0.85) * spec.h,
                         rng.uniform(0.05, 0.11) * spec.w, rng.uniform(0.06, 0.13) * spec.h,
                         rng.uniform(0.0, 3.14159)});
    }

    DensityField df;
    df.step = 8;
    df.gw = spec.w / df.step + 2;
    df.gh = spec.h / df.step + 2;
    df.v.assign(size_t(df.gw) * df.gh, 0.f);
    for (int gy = 0; gy < df.gh; ++gy) {
        for (int gx = 0; gx < df.gw; ++gx) {
            double x = gx * df.step, y = gy * df.step;
            double acc = 0.0;
            for (const Blob& b : blobs) {
                double dx = x - b.cx, dy = y - b.cy;
                double rx = dx * std::cos(b.rot) + dy * std::sin(b.rot);
                double ry = -dx * std::sin(b.rot) + dy * std::cos(b.rot);
                acc += std::exp(-(rx * rx) / (2 * b.rx * b.rx) - (ry * ry) / (2 * b.ry * b.ry));
            }
            // Irregular edges: without the warp the tissue looks like a set of
            // perfect ellipses and Otsu has an unrealistically easy job.
            acc *= 1.0 + 0.30 * std::sin(x / 190.0) * std::cos(y / 145.0) +
                   0.18 * std::sin((x + y) / 83.0);
            df.v[size_t(gy) * df.gw + gx] = float(acc);
        }
    }

    // Glass background, then stroma wherever the density field is high enough.
    const double kTissue = 0.35;
    uint64_t nstate = spec.seed ^ 0x1234;
    for (int y = 0; y < spec.h; ++y) {
        for (int x = 0; x < spec.w; ++x) {
            Rng n(nstate++);
            double d = df.sample(x, y);
            if (d < kTissue) {
                double g = 244.0 + n.normal() * 2.0;
                uint8_t v = uint8_t(std::clamp(g, 0.0, 255.0));
                img.at(x, y, 0) = v;
                img.at(x, y, 1) = v;
                img.at(x, y, 2) = uint8_t(std::clamp(g - 1.0, 0.0, 255.0));
                continue;
            }
            double t = std::min(1.0, (d - kTissue) / 0.9);
            // Eosin-stained stroma: pink, denser toward the middle of a region.
            double texture = 6.0 * std::sin(x / 3.1 + y / 5.7) + n.normal() * 4.0;
            img.at(x, y, 0) = uint8_t(std::clamp(238.0 - 40.0 * t + texture, 0.0, 255.0));
            img.at(x, y, 1) = uint8_t(std::clamp(196.0 - 60.0 * t + texture, 0.0, 255.0));
            img.at(x, y, 2) = uint8_t(std::clamp(216.0 - 35.0 * t + texture, 0.0, 255.0));
        }
    }

    // Haematoxylin nuclei. These are the high-frequency content every focus
    // metric in the project actually keys on -- a slide of flat stroma has no
    // focus signal at all.
    const long long area = 1LL * spec.w * spec.h;
    const long long n_nuclei = area / 220;
    for (long long i = 0; i < n_nuclei; ++i) {
        int x = rng.uniform_int(0, spec.w - 1);
        int y = rng.uniform_int(0, spec.h - 1);
        if (df.sample(x, y) < kTissue + 0.15) continue;
        int r = rng.uniform_int(2, 4);
        double dark = rng.uniform(0.45, 0.78);
        for (int j = -r; j <= r; ++j) {
            int py = y + j;
            if (py < 0 || py >= spec.h) continue;
            for (int k = -r; k <= r; ++k) {
                int px = x + k;
                if (px < 0 || px >= spec.w) continue;
                if (j * j + k * k > r * r) continue;
                img.at(px, py, 0) = uint8_t(img.at(px, py, 0) * dark);
                img.at(px, py, 1) = uint8_t(img.at(px, py, 1) * (dark * 0.80));
                img.at(px, py, 2) = uint8_t(std::min(255.0, img.at(px, py, 2) * (dark * 1.05)));
            }
        }
    }

    // Tissue folds: a band where the section doubled over on itself. Darker,
    // and the detail underneath is displaced.
    for (int i = 0; i < spec.folds; ++i) {
        int cx = rng.uniform_int(spec.w / 6, spec.w * 5 / 6);
        int cy = rng.uniform_int(spec.h / 6, spec.h * 5 / 6);
        int half_w = rng.uniform_int(28, 60);
        int len = rng.uniform_int(spec.h / 6, spec.h / 3);
        double ang = rng.uniform(-0.7, 0.7);
        int minx = spec.w, miny = spec.h, maxx = 0, maxy = 0;
        for (int t = -len / 2; t <= len / 2; ++t) {
            // A fold is a crease, not a ruled rectangle: let it wander.
            double wobble = 7.0 * std::sin(t / 37.0) + 4.0 * std::sin(t / 11.0);
            for (int u = -half_w; u <= half_w; ++u) {
                double uu = u + wobble;
                int px = int(cx + uu * std::cos(ang) - t * std::sin(ang));
                int py = int(cy + uu * std::sin(ang) + t * std::cos(ang));
                if (px < 0 || py < 0 || px >= spec.w || py >= spec.h) continue;
                if (df.sample(px, py) < kTissue) continue;
                int sx = std::clamp(px + 9, 0, spec.w - 1);
                int sy = std::clamp(py + 9, 0, spec.h - 1);
                for (int c = 0; c < 3; ++c) {
                    double blend = 0.5 * img.at(px, py, c) + 0.5 * img.at(sx, sy, c);
                    img.at(px, py, c) = uint8_t(std::clamp(blend * 0.68, 0.0, 255.0));
                }
                minx = std::min(minx, px); maxx = std::max(maxx, px);
                miny = std::min(miny, py); maxy = std::max(maxy, py);
            }
        }
        if (truth && maxx > minx)
            truth->artifacts.push_back({"fold", minx, miny, maxx - minx, maxy - miny, double(cx),
                                        double(cy), ang, double(half_w), double(len)});
    }

    // Mounting-medium bubbles: bright, flat interior with a dark rim.
    for (int i = 0; i < spec.bubbles; ++i) {
        int cx = rng.uniform_int(spec.w / 8, spec.w * 7 / 8);
        int cy = rng.uniform_int(spec.h / 8, spec.h * 7 / 8);
        int r = rng.uniform_int(70, 150);
        for (int j = -r; j <= r; ++j) {
            for (int k = -r; k <= r; ++k) {
                int px = cx + k, py = cy + j;
                if (px < 0 || py < 0 || px >= spec.w || py >= spec.h) continue;
                double d = std::hypot(double(k), double(j));
                if (d > r) continue;
                double rim = d > r - 6 ? 0.55 : 1.0;
                for (int c = 0; c < 3; ++c) {
                    double v = 0.25 * img.at(px, py, c) + 0.75 * 250.0;
                    img.at(px, py, c) = uint8_t(std::clamp(v * rim, 0.0, 255.0));
                }
            }
        }
        if (truth)
            truth->artifacts.push_back({"bubble", cx - r, cy - r, 2 * r, 2 * r, double(cx),
                                        double(cy), 0.0, double(r), double(2 * r)});
    }

    return img;
}

void write_truth_json(const std::string& path, const SlideSpec& spec, const SlideTruth& truth) {
    std::ofstream f(path);
    f << "{\n  \"width\": " << spec.w << ",\n  \"height\": " << spec.h << ",\n  \"seed\": "
      << spec.seed << ",\n  \"artifacts\": [\n";
    for (size_t i = 0; i < truth.artifacts.size(); ++i) {
        const Artifact& a = truth.artifacts[i];
        f << "    {\"kind\": \"" << a.kind << "\", \"x\": " << a.x << ", \"y\": " << a.y
          << ", \"w\": " << a.w << ", \"h\": " << a.h << ", \"cx\": " << a.cx
          << ", \"cy\": " << a.cy << ", \"angle\": " << a.angle << ", \"half_w\": " << a.half_w
          << ", \"length\": " << a.length << "}";
        f << (i + 1 < truth.artifacts.size() ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
}

}  // namespace wsi
