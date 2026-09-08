// Focus metrics. Both are computed on gray and normalised per pixel so that
// frames of different sizes (the AF ROI vs a full FOV) stay comparable.
#pragma once
#include "wsi/image.hpp"

namespace wsi {

// Brenner gradient: mean squared difference at a fixed lag. Cheap, and the
// classic choice for brightfield microscopy autofocus.
double brenner(const Image& gray, int lag = 2);

// Variance of the Laplacian: sharper peak, but noisier far from focus.
double var_laplacian(const Image& gray);

enum class FocusMetric { Brenner, VarLaplacian };
double focus_score(const Image& gray, FocusMetric m);
const char* metric_name(FocusMetric m);

}  // namespace wsi
