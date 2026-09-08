#include "wsi/camera.hpp"

#include <algorithm>
#include <cmath>

#include "wsi/rng.hpp"

namespace wsi {

SyntheticStage::SyntheticStage(double speed_px_per_ms, double settle_ms, double z_ms_per_um,
                               double z_settle_ms)
    : speed_(speed_px_per_ms), settle_(settle_ms), z_rate_(z_ms_per_um), z_settle_(z_settle_ms) {}

void SyntheticStage::move_xy(int x, int y) {
    if (x == x_ && y == y_) return;
    double d = std::hypot(double(x - x_), double(y - y_));
    x_ = x;
    y_ = y;
    st_.xy_moves++;
    st_.travel_px += d;
    st_.move_time_ms += settle_ + d / speed_;
}

void SyntheticStage::move_z(double z_um) {
    double d = std::abs(z_um - z_);
    if (d < 1e-9) return;
    z_ = z_um;
    st_.z_moves++;
    st_.travel_z_um += d;
    st_.move_time_ms += z_settle_ + d * z_rate_;
}

SyntheticCamera::SyntheticCamera(const Image& slide, const IStage& stage, OpticsParams p)
    : slide_(slide), stage_(stage), p_(p), noise_state_(p.seed * 0x2545F4914F6CDD1DULL + 11) {}

double SyntheticCamera::true_focus_z(int x, int y) const {
    // Best-focus height at the centre of the FOV whose top-left is (x, y).
    double cx = x + p_.fov_w * 0.5, cy = y + p_.fov_h * 0.5;
    double z = p_.z_offset + p_.z_tilt_x * cx + p_.z_tilt_y * cy;
    z += p_.z_bow * std::sin(6.283185307179586 * cx / p_.z_bow_period) *
         std::cos(6.283185307179586 * cy / (p_.z_bow_period * 1.37));
    if (p_.anomaly_r > 0) {
        double d = std::hypot(cx - p_.anomaly_x, cy - p_.anomaly_y);
        if (d < p_.anomaly_r) {
            // A raised patch with a soft edge: locally real, globally invisible
            // to a plane fit.
            double t = 0.5 * (1.0 + std::cos(3.14159265358979 * d / p_.anomaly_r));
            z += p_.anomaly_dz * t;
        }
    }
    return z;
}

std::pair<int, int> SyntheticCamera::true_offset(int x, int y) const {
    if (p_.xy_jitter_px <= 0) return {0, 0};
    uint64_t h = hash_site(p_.seed, x, y);
    int span = int(p_.xy_jitter_px);
    int jx = int(h % uint64_t(2 * span + 1)) - span;
    int jy = int((h >> 32) % uint64_t(2 * span + 1)) - span;
    return {jx, jy};
}

Image SyntheticCamera::render(int out_w, int out_h, int bias_x, int bias_y) {
    const int sx = stage_.x(), sy = stage_.y();
    auto [jx, jy] = true_offset(sx, sy);

    Image frame = crop(slide_, sx + jx + bias_x, sy + jy + bias_y, out_w, out_h);

    double dz = std::abs(stage_.z() - true_focus_z(sx, sy));
    double sigma = std::min(p_.blur_cap, dz * p_.blur_px_per_um);
    blur_gaussian(frame, float(sigma));

    // Vignette is a property of the optics, so it is anchored to the full FOV
    // even when only a centred ROI is read out.
    const double fw = p_.fov_w, fh = p_.fov_h;
    const double ox = (fw - out_w) * 0.5 + bias_x, oy = (fh - out_h) * 0.5 + bias_y;
    const double cx = fw * 0.5, cy = fh * 0.5;
    const double rmax2 = cx * cx + cy * cy;

    uint64_t gh = hash_site(p_.seed ^ 0xABCDEF, sx, sy);
    double gain = 1.0 + p_.illum_gain_jitter * ((double(gh >> 40) / double(1 << 24)) - 0.5) * 2.0;

    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            double px = ox + x - cx, py = oy + y - cy;
            double v = 1.0 - p_.vignette * ((px * px + py * py) / rmax2);
            double shade = v * gain;
            for (int c = 0; c < frame.ch; ++c) {
                double val = frame.at(x, y, c) * shade;
                if (p_.noise_sigma > 0) {
                    Rng r(noise_state_++);
                    val += r.normal() * p_.noise_sigma;
                }
                frame.at(x, y, c) = uint8_t(std::clamp(val + 0.5, 0.0, 255.0));
            }
        }
    }

    st_.exposures++;
    st_.pixels += uint64_t(out_w) * out_h;
    st_.exposure_time_ms += p_.exposure_ms;
    return frame;
}

Image SyntheticCamera::grab() { return render(p_.fov_w, p_.fov_h, 0, 0); }

Image SyntheticCamera::grab_roi(int w, int h) {
    w = std::min(w, p_.fov_w);
    h = std::min(h, p_.fov_h);
    return render(w, h, 0, 0);
}

Image SyntheticCamera::blank_frame() const {
    // White light through empty glass: uniform illumination shaped by the same
    // vignette the real frames get, with a little noise.
    Image f(p_.fov_w, p_.fov_h, 3);
    const double cx = p_.fov_w * 0.5, cy = p_.fov_h * 0.5;
    const double rmax2 = cx * cx + cy * cy;
    uint64_t state = p_.seed ^ 0x5151;
    for (int y = 0; y < f.h; ++y) {
        for (int x = 0; x < f.w; ++x) {
            double px = x - cx, py = y - cy;
            double v = 1.0 - p_.vignette * ((px * px + py * py) / rmax2);
            for (int c = 0; c < 3; ++c) {
                Rng r(state++);
                double val = 250.0 * v + r.normal() * p_.noise_sigma;
                f.at(x, y, c) = uint8_t(std::clamp(val + 0.5, 0.0, 255.0));
            }
        }
    }
    return f;
}

}  // namespace wsi
