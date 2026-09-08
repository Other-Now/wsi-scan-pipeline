#include "wsi/scan.hpp"

#include <algorithm>
#include <cmath>

namespace wsi {

int ScanPlan::index_of(int col, int row) const {
    for (size_t i = 0; i < tiles.size(); ++i)
        if (tiles[i].col == col && tiles[i].row == row) return int(i);
    return -1;
}

ScanPlan plan_serpentine(const TissueSeg& seg, int slide_w, int slide_h, int fov_w, int fov_h,
                         int overlap, double min_coverage) {
    ScanPlan plan;
    plan.overlap = overlap;
    plan.step_x = fov_w - overlap;
    plan.step_y = fov_h - overlap;
    if (plan.step_x <= 0 || plan.step_y <= 0) return plan;

    const int x0 = std::clamp(seg.x0, 0, std::max(0, slide_w - 1));
    const int y0 = std::clamp(seg.y0, 0, std::max(0, slide_h - 1));
    const int x1 = std::clamp(seg.x1, x0 + 1, slide_w);
    const int y1 = std::clamp(seg.y1, y0 + 1, slide_h);

    plan.cols = std::max(1, (x1 - x0 + plan.step_x - 1) / plan.step_x);
    plan.rows = std::max(1, (y1 - y0 + plan.step_y - 1) / plan.step_y);

    for (int row = 0; row < plan.rows; ++row) {
        for (int i = 0; i < plan.cols; ++i) {
            int col = (row % 2 == 0) ? i : plan.cols - 1 - i;  // serpentine
            ScanTile t;
            t.col = col;
            t.row = row;
            t.x = x0 + col * plan.step_x;
            t.y = y0 + row * plan.step_y;
            t.tissue_cov = seg.mask.coverage(t.x, t.y, fov_w, fov_h);
            t.acquire = t.tissue_cov >= min_coverage;
            plan.tiles.push_back(t);
            plan.total++;
            if (t.acquire) plan.to_acquire++;
        }
    }
    return plan;
}

void FocusMap::observe(double x, double y, double z) {
    xs_.push_back(x);
    ys_.push_back(y);
    zs_.push_back(z);
    ++n_;
    refit();
}

void FocusMap::refit() {
    if (n_ < 3) return;
    // Normal equations for z = a + b*x + c*y.
    double s1 = double(n_), sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sz = 0, sxz = 0, syz = 0;
    for (size_t i = 0; i < n_; ++i) {
        double x = xs_[i], y = ys_[i], z = zs_[i];
        sx += x; sy += y; sz += z;
        sxx += x * x; syy += y * y; sxy += x * y;
        sxz += x * z; syz += y * z;
    }
    double m[3][3] = {{s1, sx, sy}, {sx, sxx, sxy}, {sy, sxy, syy}};
    double v[3] = {sz, sxz, syz};
    auto det3 = [](double a[3][3]) {
        return a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
               a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
               a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    };
    double d = det3(m);
    if (std::abs(d) < 1e-9) return;
    double out[3];
    for (int k = 0; k < 3; ++k) {
        double t[3][3];
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) t[r][c] = (c == k) ? v[r] : m[r][c];
        out[k] = det3(t) / d;
    }
    a_ = out[0];
    b_ = out[1];
    c_ = out[2];
    fitted_ = true;
}

double FocusMap::predict(double x, double y) const {
    double base = a_ + b_ * x + c_ * y;
    if (mode_ == Mode::Plane || n_ == 0) return base;

    // Inverse-distance-weighted residual. The +r^2 in the denominator makes the
    // correction fade to the bare plane as you move away from anything that has
    // actually been measured, instead of extrapolating a neighbour's bow across
    // the slide.
    const double r2 = local_radius_ * local_radius_;
    double wsum = 0, acc = 0;
    for (size_t i = 0; i < n_; ++i) {
        double dx = x - xs_[i], dy = y - ys_[i];
        double w = 1.0 / (dx * dx + dy * dy + r2);
        acc += w * (zs_[i] - (a_ + b_ * xs_[i] + c_ * ys_[i]));
        wsum += w;
    }
    return wsum > 0 ? base + acc / wsum : base;
}

double FocusMap::rms_residual() const {
    if (!fitted_ || n_ == 0) return 0.0;
    double acc = 0;
    for (size_t i = 0; i < n_; ++i) {
        double r = zs_[i] - predict(xs_[i], ys_[i]);
        acc += r * r;
    }
    return std::sqrt(acc / double(n_));
}

namespace {

double score_at(IStage& stage, ICamera& cam, double z, const AfConfig& cfg, int& frames) {
    stage.move_z(z);
    Image roi = cam.grab_roi(cfg.roi, cfg.roi);
    ++frames;
    return focus_score(to_gray(roi), cfg.metric);
}

}  // namespace

AfResult autofocus_sweep(IStage& stage, ICamera& cam, const AfConfig& cfg, double z_centre) {
    AfResult r;
    r.full_sweep = true;

    const double z_lo = z_centre - cfg.sweep_half_range;
    const double z_hi = z_centre + cfg.sweep_half_range;
    double best_z = z_lo, best_s = -1.0;
    for (double z = z_lo; z <= z_hi + 1e-9; z += cfg.coarse_step) {
        double s = score_at(stage, cam, z, cfg, r.frames);
        if (s > best_s) {
            best_s = s;
            best_z = z;
        }
    }

    // Fine hill-climb from the coarse winner. Walking downhill once and stopping
    // is enough because the metric is unimodal near focus; the coarse pass is
    // what protects against the far-from-focus plateau.
    double z = best_z, s = best_s;
    for (int dir : {+1, -1}) {
        int steps = 0;
        double zc = z, sc = s;
        while (steps++ < cfg.max_fine_steps) {
            double zn = zc + dir * cfg.fine_step;
            if (zn < z_lo - cfg.coarse_step || zn > z_hi + cfg.coarse_step) break;
            double sn = score_at(stage, cam, zn, cfg, r.frames);
            if (sn <= sc) break;
            zc = zn;
            sc = sn;
        }
        if (sc > best_s) {
            best_s = sc;
            best_z = zc;
        }
    }

    stage.move_z(best_z);
    r.z = best_z;
    r.score = best_s;
    return r;
}

AfResult autofocus_with_map(IStage& stage, ICamera& cam, const FocusMap& map, const AfConfig& cfg,
                            double expected_score) {
    AfResult r;
    r.full_sweep = false;
    double zp = map.predict(double(stage.x()), double(stage.y()));

    // --trust-map: take the prediction and go. One frame per tile, and no way
    // to notice that the coverslip is not where the map says it is. This exists
    // to show what the verification is buying, and what the tile-quality model
    // has to catch when there is none.
    if (!cfg.verify) {
        r.score = score_at(stage, cam, zp, cfg, r.frames);
        r.z = zp;
        return r;
    }

    // Three frames: below, at, and above the prediction. The test is the *shape*
    // of that triple -- a concave parabola whose vertex falls inside the span.
    // Shape, not an absolute score, because how high the score gets depends on
    // how much tissue this particular tile happens to contain; a fixed
    // threshold would fire on every sparse tile at the edge of a section.
    double s_lo = score_at(stage, cam, zp - cfg.verify_span, cfg, r.frames);
    double s_mid = score_at(stage, cam, zp, cfg, r.frames);
    double s_hi = score_at(stage, cam, zp + cfg.verify_span, cfg, r.frames);

    const double denom = s_lo - 2.0 * s_mid + s_hi;
    const bool concave = denom < 0.0;
    const double frac = concave ? 0.5 * (s_lo - s_hi) / denom : 0.0;  // in units of verify_span
    const bool has_signal = expected_score <= 0.0 || s_mid > cfg.accept_ratio * 0.1 * expected_score;

    // Where the slide is not the surface the map thinks it is -- debris, a
    // bubble lifting the coverslip -- the prediction is simply wrong, and the
    // honest response is to pay for a full sweep rather than record a blurred
    // tile and move on.
    if (!concave || std::abs(frac) > 1.0 || !has_signal) {
        AfResult full = autofocus_sweep(stage, cam, cfg, zp);
        full.frames += r.frames;
        full.full_sweep = true;
        return full;
    }

    // One confirming frame at the interpolated vertex. If the parabola lied,
    // the middle sample is still a measured, in-bracket result to fall back on.
    double z_star = zp + frac * cfg.verify_span;
    double s_star = score_at(stage, cam, z_star, cfg, r.frames);
    if (s_star >= s_mid) {
        r.z = z_star;
        r.score = s_star;
    } else {
        r.z = zp;
        r.score = s_mid;
    }
    stage.move_z(r.z);
    return r;
}

}  // namespace wsi
