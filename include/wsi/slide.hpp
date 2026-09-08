// A procedural H&E-like slide, used as the ground truth the synthetic camera
// crops FOVs out of.
//
// Why generate one instead of downloading a real .svs: every number this
// project reports is an error against a known truth -- where the tissue is,
// where the stage actually landed, which tiles carry a fold or a bubble. A real
// slide has none of that written down. Real data still works (see
// python/make_slide.py, which converts any image into this format), it just
// cannot be scored the same way.
#pragma once
#include <string>
#include <vector>

#include "wsi/image.hpp"

namespace wsi {

struct Artifact {
    std::string kind;             // "fold" or "bubble"
    int x = 0, y = 0, w = 0, h = 0;  // bounding box
    // The shape itself, so a scoring pass can ask "is this patch on the fold?"
    // rather than "does it touch the fold's bounding box?" -- the bounding box
    // of a diagonal band is mostly not the band.
    double cx = 0, cy = 0;        // centre
    double angle = 0;             // fold only: band direction, radians
    double half_w = 0;            // fold half-width, or bubble radius
    double length = 0;            // fold length, or bubble diameter
};

struct SlideTruth {
    std::vector<Artifact> artifacts;
};

struct SlideSpec {
    int w = 6144, h = 4096;
    uint64_t seed = 42;
    int blobs = 5;    // tissue regions
    int folds = 3;    // tissue folds (a real scanner artefact: doubled, darker)
    int bubbles = 4;  // mounting-medium bubbles (bright, dark rim)
};

Image generate_slide(const SlideSpec& spec, SlideTruth* truth = nullptr);
void write_truth_json(const std::string& path, const SlideSpec& spec, const SlideTruth& truth);

}  // namespace wsi
