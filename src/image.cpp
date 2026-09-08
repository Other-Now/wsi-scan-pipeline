#include "wsi/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace wsi {

Image crop(const Image& src, int x, int y, int w, int h) {
    Image out(w, h, src.ch);
    for (int j = 0; j < h; ++j) {
        int sy = std::clamp(y + j, 0, src.h - 1);
        for (int i = 0; i < w; ++i) {
            int sx = std::clamp(x + i, 0, src.w - 1);
            std::memcpy(&out.px[out.idx(i, j)], &src.px[src.idx(sx, sy)], src.ch);
        }
    }
    return out;
}

Image to_gray(const Image& src) {
    if (src.ch == 1) return src;
    Image out(src.w, src.h, 1);
    for (size_t i = 0, n = size_t(src.w) * src.h; i < n; ++i) {
        const uint8_t* p = &src.px[i * src.ch];
        // Rec.601 luma in fixed point; H&E background is bright, tissue is dark.
        out.px[i] = uint8_t((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
    }
    return out;
}

// The pyramid spends all of its time here, so 2x gets its own loop: row
// pointers instead of an index computation per sample, and the channel loop
// unrolled by the compiler because ch is loop-invariant. Bit-identical to
// downsample(src, 2) -- the unit test asserts that, not just that it is fast.
Image downsample2(const Image& src) {
    const int w = src.w / 2, h = src.h / 2, ch = src.ch;
    if (w < 1 || h < 1) return downsample(src, 2);
    Image out(w, h, ch);
    for (int y = 0; y < h; ++y) {
        const uint8_t* r0 = &src.px[src.idx(0, 2 * y)];
        const uint8_t* r1 = &src.px[src.idx(0, 2 * y + 1)];
        uint8_t* d = &out.px[out.idx(0, y)];
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c)
                d[c] = uint8_t((unsigned(r0[c]) + r0[ch + c] + r1[c] + r1[ch + c]) >> 2);
            r0 += 2 * ch;
            r1 += 2 * ch;
            d += ch;
        }
    }
    return out;
}

Image downsample(const Image& src, int f) {
    if (f <= 1) return src;
    int w = std::max(1, src.w / f), h = std::max(1, src.h / f);
    Image out(w, h, src.ch);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < src.ch; ++c) {
                unsigned acc = 0, n = 0;
                for (int j = 0; j < f; ++j) {
                    int sy = y * f + j;
                    if (sy >= src.h) break;
                    for (int i = 0; i < f; ++i) {
                        int sx = x * f + i;
                        if (sx >= src.w) break;
                        acc += src.at(sx, sy, c);
                        ++n;
                    }
                }
                out.at(x, y, c) = uint8_t(acc / std::max(1u, n));
            }
        }
    }
    return out;
}

void blur_gaussian(Image& img, float sigma) {
    if (sigma <= 0.05f) return;
    int r = std::max(1, int(std::ceil(3.0f * sigma)));
    std::vector<float> k(size_t(2 * r + 1));
    float sum = 0.f;
    for (int i = -r; i <= r; ++i) {
        k[size_t(i + r)] = std::exp(-float(i * i) / (2.f * sigma * sigma));
        sum += k[size_t(i + r)];
    }
    for (auto& v : k) v /= sum;

    const int w = img.w, h = img.h, ch = img.ch;
    std::vector<uint8_t> tmp(img.px.size());
    // horizontal
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float acc = 0.f;
                for (int i = -r; i <= r; ++i) {
                    int sx = std::clamp(x + i, 0, w - 1);
                    acc += k[size_t(i + r)] * img.at(sx, y, c);
                }
                tmp[img.idx(x, y, c)] = uint8_t(std::clamp(acc + 0.5f, 0.f, 255.f));
            }
        }
    }
    // vertical
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ch; ++c) {
                float acc = 0.f;
                for (int i = -r; i <= r; ++i) {
                    int sy = std::clamp(y + i, 0, h - 1);
                    acc += k[size_t(i + r)] * tmp[img.idx(x, sy, c)];
                }
                img.at(x, y, c) = uint8_t(std::clamp(acc + 0.5f, 0.f, 255.f));
            }
        }
    }
}

static void skip_ws_comments(std::istream& is) {
    for (;;) {
        int c = is.peek();
        if (c == '#') { std::string l; std::getline(is, l); }
        else if (c == ' ' || c == '\n' || c == '\r' || c == '\t') is.get();
        else return;
    }
}

Image read_pnm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string magic;
    f >> magic;
    if (magic != "P5" && magic != "P6") throw std::runtime_error("not a P5/P6 PNM: " + path);
    int w, h, maxv;
    skip_ws_comments(f); f >> w;
    skip_ws_comments(f); f >> h;
    skip_ws_comments(f); f >> maxv;
    f.get();
    if (maxv != 255) throw std::runtime_error("only 8-bit PNM supported");
    Image img(w, h, magic == "P5" ? 1 : 3);
    f.read(reinterpret_cast<char*>(img.px.data()), std::streamsize(img.px.size()));
    if (size_t(f.gcount()) != img.px.size()) throw std::runtime_error("truncated PNM: " + path);
    return img;
}

void write_pnm(const std::string& path, const Image& img) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path);
    f << (img.ch == 1 ? "P5\n" : "P6\n") << img.w << " " << img.h << "\n255\n";
    f.write(reinterpret_cast<const char*>(img.px.data()), std::streamsize(img.px.size()));
}

}  // namespace wsi
