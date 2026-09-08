// Tissue-vs-background segmentation (Otsu) and flat-field correction.
#pragma once
#include <cstdint>
#include <vector>

#include "wsi/image.hpp"

namespace wsi {

struct TissueMask {
    int w = 0, h = 0;  // mask dimensions, at thumbnail scale
    int scale = 1;     // slide pixels per mask pixel
    std::vector<uint8_t> v;

    bool at(int mx, int my) const {
        if (mx < 0 || my < 0 || mx >= w || my >= h) return false;
        return v[size_t(my) * w + mx] != 0;
    }
    // Fraction of a slide-space rectangle covered by tissue.
    double coverage(int x, int y, int rw, int rh) const;
};

struct TissueSeg {
    TissueMask mask;
    int threshold = 0;       // Otsu threshold on the luminance histogram
    double tissue_frac = 0;  // fraction of the whole slide that is tissue
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // tissue bounding box, slide coordinates
};

int otsu_threshold(const uint64_t hist[256]);
TissueSeg segment_tissue(const Image& slide, int thumb_scale);
Image mask_to_image(const TissueMask& m);

// Illumination correction estimated from a blank (tissue-free) frame.
struct FlatField {
    int w = 0, h = 0, ch = 0;
    std::vector<float> gain;  // multiply the raw frame by this
    bool valid() const { return !gain.empty(); }
};

FlatField estimate_flatfield(const Image& blank);
void apply_flatfield(Image& img, const FlatField& ff);

}  // namespace wsi
