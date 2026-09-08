#include "wsi/registration.hpp"

#include <algorithm>
#include <cmath>

namespace wsi {

static bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

void fft1d(std::vector<std::complex<float>>& a, bool inverse) {
    const size_t n = a.size();
    if (n < 2) return;
    // bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * 3.14159265358979323846 / double(len) * (inverse ? 1.0 : -1.0);
        std::complex<float> wl(float(std::cos(ang)), float(std::sin(ang)));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.f, 0.f);
            for (size_t k = 0; k < len / 2; ++k) {
                std::complex<float> u = a[i + k];
                std::complex<float> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wl;
            }
        }
    }
    if (inverse)
        for (auto& v : a) v /= float(n);
}

void fft2d(std::vector<std::complex<float>>& a, int w, int h, bool inverse) {
    std::vector<std::complex<float>> row(static_cast<size_t>(w));
    std::vector<std::complex<float>> col(static_cast<size_t>(h));
    for (int y = 0; y < h; ++y) {
        std::copy(a.begin() + size_t(y) * w, a.begin() + size_t(y + 1) * w, row.begin());
        fft1d(row, inverse);
        std::copy(row.begin(), row.end(), a.begin() + size_t(y) * w);
    }
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) col[size_t(y)] = a[size_t(y) * w + x];
        fft1d(col, inverse);
        for (int y = 0; y < h; ++y) a[size_t(y) * w + x] = col[size_t(y)];
    }
}

// Parabolic interpolation through the peak and its two neighbours, which buys
// sub-pixel accuracy without a second correlation pass.
static float subpixel(float left, float centre, float right) {
    float denom = left - 2.f * centre + right;
    if (std::abs(denom) < 1e-6f) return 0.f;
    float d = 0.5f * (left - right) / denom;
    return std::clamp(d, -1.f, 1.f);
}

Shift phase_correlate(const Image& a, const Image& b) {
    Shift s;
    if (a.ch != 1 || b.ch != 1 || a.w != b.w || a.h != b.h) return s;
    if (!is_pow2(a.w) || !is_pow2(a.h)) return s;

    const int w = a.w, h = a.h;
    const size_t n = size_t(w) * h;

    // Hann window in both axes: without it the frame edges dominate the
    // spectrum and the correlation peak lands on 0,0 no matter the real shift.
    std::vector<float> wx(static_cast<size_t>(w));
    std::vector<float> wy(static_cast<size_t>(h));
    for (int x = 0; x < w; ++x)
        wx[size_t(x)] = 0.5f * (1.f - std::cos(2.f * 3.14159265f * float(x) / float(w - 1)));
    for (int y = 0; y < h; ++y)
        wy[size_t(y)] = 0.5f * (1.f - std::cos(2.f * 3.14159265f * float(y) / float(h - 1)));

    std::vector<std::complex<float>> fa(n), fb(n);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float win = wx[size_t(x)] * wy[size_t(y)];
            fa[size_t(y) * w + x] = std::complex<float>(float(a.at(x, y)) * win, 0.f);
            fb[size_t(y) * w + x] = std::complex<float>(float(b.at(x, y)) * win, 0.f);
        }
    }
    fft2d(fa, w, h, false);
    fft2d(fb, w, h, false);

    // Cross-power spectrum, magnitude-normalised: only the phase difference
    // survives, which is why this is robust to the illumination difference
    // between two neighbouring FOVs.
    for (size_t i = 0; i < n; ++i) {
        std::complex<float> r = fa[i] * std::conj(fb[i]);
        float m = std::abs(r);
        fa[i] = m > 1e-9f ? r / m : std::complex<float>(0.f, 0.f);
    }
    fft2d(fa, w, h, true);

    size_t best = 0;
    float best_v = -1.f, sum = 0.f;
    for (size_t i = 0; i < n; ++i) {
        float v = fa[i].real();
        sum += std::abs(v);
        if (v > best_v) {
            best_v = v;
            best = i;
        }
    }
    int px = int(best % size_t(w)), py = int(best / size_t(w));

    auto real_at = [&](int x, int y) {
        x = ((x % w) + w) % w;
        y = ((y % h) + h) % h;
        return fa[size_t(y) * w + x].real();
    };
    float sx = subpixel(real_at(px - 1, py), best_v, real_at(px + 1, py));
    float sy = subpixel(real_at(px, py - 1), best_v, real_at(px, py + 1));

    // The correlation surface is periodic: a peak past the halfway point is a
    // negative shift.
    int ix = px > w / 2 ? px - w : px;
    int iy = py > h / 2 ? py - h : py;

    s.dx = float(ix) + sx;
    s.dy = float(iy) + sy;
    s.peak = sum > 0.f ? best_v / (sum / float(n)) : 0.f;  // peak over mean magnitude
    s.ok = true;
    return s;
}

}  // namespace wsi
