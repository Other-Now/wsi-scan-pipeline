#include "wsi/segment.hpp"

#include <algorithm>
#include <cmath>

namespace wsi {

double TissueMask::coverage(int x, int y, int rw, int rh) const {
    int mx0 = x / scale, my0 = y / scale;
    int mx1 = (x + rw + scale - 1) / scale, my1 = (y + rh + scale - 1) / scale;
    mx0 = std::clamp(mx0, 0, w);
    my0 = std::clamp(my0, 0, h);
    mx1 = std::clamp(mx1, 0, w);
    my1 = std::clamp(my1, 0, h);
    long long total = 0, on = 0;
    for (int my = my0; my < my1; ++my) {
        for (int mx = mx0; mx < mx1; ++mx) {
            ++total;
            on += at(mx, my) ? 1 : 0;
        }
    }
    return total ? double(on) / double(total) : 0.0;
}

int otsu_threshold(const uint64_t hist[256]) {
    uint64_t total = 0;
    double sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        total += hist[i];
        sum += double(i) * double(hist[i]);
    }
    if (!total) return 128;

    double sum_b = 0.0, best_var = -1.0;
    uint64_t w_b = 0;
    int best = 128;
    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (!w_b) continue;
        uint64_t w_f = total - w_b;
        if (!w_f) break;
        sum_b += double(t) * double(hist[t]);
        double m_b = sum_b / double(w_b);
        double m_f = (sum - sum_b) / double(w_f);
        double var = double(w_b) * double(w_f) * (m_b - m_f) * (m_b - m_f);
        if (var > best_var) {
            best_var = var;
            best = t;
        }
    }
    return best;
}

// 3x3 binary morphology on the thumbnail-scale mask.
static std::vector<uint8_t> morph(const std::vector<uint8_t>& in, int w, int h, bool dilate) {
    std::vector<uint8_t> out(in.size(), 0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            bool hit = !dilate;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int sx = std::clamp(x + i, 0, w - 1);
                    int sy = std::clamp(y + j, 0, h - 1);
                    bool v = in[size_t(sy) * w + sx] != 0;
                    if (dilate && v) hit = true;
                    if (!dilate && !v) hit = false;
                }
            }
            out[size_t(y) * w + x] = uint8_t(hit ? 255 : 0);
        }
    }
    return out;
}

TissueSeg segment_tissue(const Image& slide, int thumb_scale) {
    TissueSeg seg;
    Image thumb = to_gray(downsample(slide, thumb_scale));

    uint64_t hist[256] = {0};
    for (uint8_t p : thumb.px) hist[p]++;
    seg.threshold = otsu_threshold(hist);

    std::vector<uint8_t> m(thumb.px.size());
    for (size_t i = 0; i < thumb.px.size(); ++i)
        m[i] = uint8_t(thumb.px[i] <= seg.threshold ? 255 : 0);  // tissue is darker than glass

    // open (drop specks and dust), then close (fill pinholes inside tissue)
    m = morph(m, thumb.w, thumb.h, false);
    m = morph(m, thumb.w, thumb.h, true);
    m = morph(m, thumb.w, thumb.h, true);
    m = morph(m, thumb.w, thumb.h, false);

    seg.mask = TissueMask{thumb.w, thumb.h, thumb_scale, m};

    long long on = 0;
    int x0 = thumb.w, y0 = thumb.h, x1 = 0, y1 = 0;
    for (int y = 0; y < thumb.h; ++y) {
        for (int x = 0; x < thumb.w; ++x) {
            if (!m[size_t(y) * thumb.w + x]) continue;
            ++on;
            x0 = std::min(x0, x);
            y0 = std::min(y0, y);
            x1 = std::max(x1, x);
            y1 = std::max(y1, y);
        }
    }
    seg.tissue_frac = double(on) / double(thumb.px.size());
    if (on == 0) {
        seg.x1 = slide.w;
        seg.y1 = slide.h;
        return seg;
    }
    seg.x0 = x0 * thumb_scale;
    seg.y0 = y0 * thumb_scale;
    seg.x1 = std::min(slide.w, (x1 + 1) * thumb_scale);
    seg.y1 = std::min(slide.h, (y1 + 1) * thumb_scale);
    return seg;
}

Image mask_to_image(const TissueMask& m) {
    Image img(m.w, m.h, 1);
    img.px = m.v;
    return img;
}

FlatField estimate_flatfield(const Image& blank) {
    FlatField ff;
    ff.w = blank.w;
    ff.h = blank.h;
    ff.ch = blank.ch;
    ff.gain.resize(blank.px.size());
    // Reference level is the mean of the blank frame, per channel. The gain that
    // restores every pixel to that level removes both vignette and any tilt in
    // the illumination.
    for (int c = 0; c < blank.ch; ++c) {
        double sum = 0.0;
        for (int y = 0; y < blank.h; ++y)
            for (int x = 0; x < blank.w; ++x) sum += blank.at(x, y, c);
        double mean = sum / (double(blank.w) * blank.h);
        for (int y = 0; y < blank.h; ++y) {
            for (int x = 0; x < blank.w; ++x) {
                double v = std::max(1.0, double(blank.at(x, y, c)));
                ff.gain[blank.idx(x, y, c)] = float(std::clamp(mean / v, 0.25, 4.0));
            }
        }
    }
    return ff;
}

void apply_flatfield(Image& img, const FlatField& ff) {
    if (!ff.valid() || ff.w != img.w || ff.h != img.h || ff.ch != img.ch) return;
    for (size_t i = 0; i < img.px.size(); ++i)
        img.px[i] = uint8_t(std::clamp(double(img.px[i]) * ff.gain[i] + 0.5, 0.0, 255.0));
}

}  // namespace wsi
