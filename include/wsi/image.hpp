// Minimal 8-bit image container plus the pixel ops the pipeline needs.
// Deliberately dependency-free: PNM in/out is enough to move data between the
// C++ scanner and the Python tooling (Pillow reads PNM), and the only image
// format this project *writes* for real is the pyramidal TIFF in tiff.hpp.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace wsi {

struct Image {
    int w = 0, h = 0, ch = 0;          // ch is 1 (gray) or 3 (RGB), interleaved
    std::vector<uint8_t> px;

    Image() = default;
    Image(int w_, int h_, int ch_) : w(w_), h(h_), ch(ch_), px(size_t(w_) * h_ * ch_, 0) {}

    size_t idx(int x, int y, int c = 0) const { return (size_t(y) * w + x) * ch + c; }
    uint8_t& at(int x, int y, int c = 0) { return px[idx(x, y, c)]; }
    uint8_t at(int x, int y, int c = 0) const { return px[idx(x, y, c)]; }
    size_t bytes() const { return px.size(); }
    bool empty() const { return px.empty(); }
};

// Border pixels are replicated, so a crop that runs off the slide is still a
// full-size frame -- the scanner never has to special-case edge FOVs.
Image crop(const Image& src, int x, int y, int w, int h);
Image to_gray(const Image& src);
Image downsample2(const Image& src);              // 2x2 box
Image downsample(const Image& src, int factor);   // NxN box, for thumbnails
void  blur_gaussian(Image& img, float sigma);     // separable, in place

Image read_pnm(const std::string& path);
void  write_pnm(const std::string& path, const Image& img);

}  // namespace wsi
