// Scan-path planning and autofocus.
#pragma once
#include <cstddef>
#include <vector>

#include "wsi/camera.hpp"
#include "wsi/focus.hpp"
#include "wsi/segment.hpp"

namespace wsi {

struct ScanTile {
    int col = 0, row = 0;
    int x = 0, y = 0;          // FOV top-left in slide pixels
    double tissue_cov = 0.0;   // fraction of this FOV covered by tissue
    bool acquire = false;
};

struct ScanPlan {
    std::vector<ScanTile> tiles;  // serpentine order over the tissue bounding box
    int cols = 0, rows = 0;
    int step_x = 0, step_y = 0, overlap = 0;
    int total = 0, to_acquire = 0;

    // Index into `tiles` for a grid position, or -1.
    int index_of(int col, int row) const;
};

// Serpentine raster over the tissue bounding box: odd rows run right-to-left so
// the stage never flies back across the slide at the end of a row.
ScanPlan plan_serpentine(const TissueSeg& seg, int slide_w, int slide_h, int fov_w, int fov_h,
                         int overlap, double min_coverage = 0.02);

struct AfConfig {
    // The sweep is centred on the best estimate available -- the focus map, or
    // the previous tile -- not on absolute zero. Across a 6144 px slide the
    // focal surface here moves ~16 um, so a window fixed around zero simply
    // does not contain the far corner's focus.
    double sweep_half_range = 12.0;    // um either side of the centre
    double coarse_step = 2.0;
    double fine_step = 0.4;
    int roi = 192;                     // autofocus reads out a centred ROI, not the full FOV
    FocusMetric metric = FocusMetric::Brenner;
    bool verify = true;                // check the map's prediction before trusting it
    double verify_span = 2.0;          // +/- around the predicted z when a map exists
    double accept_ratio = 0.55;        // of the expected in-focus score
    int max_fine_steps = 8;
};

struct AfResult {
    double z = 0;
    double score = 0;
    int frames = 0;         // camera exposures spent
    bool full_sweep = true; // false when the focus map carried the tile
};

// A slide-wide focus map, built from the tiles that were focused the hard way.
//
// Plane: a least-squares plane, which is the honest model of a slide clamped on
// a tilted stage. PlaneLocal adds an inverse-distance-weighted correction from
// nearby observations, which is what actually pays for itself here -- a real
// coverslip bows, and the residual after the plane fit is spatially smooth
// rather than random.
class FocusMap {
public:
    enum class Mode { Plane, PlaneLocal };

    void set_mode(Mode m) { mode_ = m; }
    Mode mode() const { return mode_; }
    void set_local_radius(double r) { local_radius_ = r; }

    void observe(double x, double y, double z);
    bool ready(size_t min_points = 6) const { return n_ >= min_points && fitted_; }
    double predict(double x, double y) const;
    double rms_residual() const;  // of the plane fit alone
    size_t size() const { return n_; }

private:
    void refit();
    Mode mode_ = Mode::PlaneLocal;
    double local_radius_ = 800.0;  // slide px; the weight halves around here
    size_t n_ = 0;
    bool fitted_ = false;
    double a_ = 0, b_ = 0, c_ = 0;  // z = a + b*x + c*y
    std::vector<double> xs_, ys_, zs_;
};

AfResult autofocus_sweep(IStage& stage, ICamera& cam, const AfConfig& cfg, double z_centre = 0.0);
AfResult autofocus_with_map(IStage& stage, ICamera& cam, const FocusMap& map, const AfConfig& cfg,
                            double expected_score);

}  // namespace wsi
